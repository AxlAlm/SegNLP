
#basics
import numpy as np
import warnings
import os
import pandas as pd
from typing import List, Dict, Union, Tuple
import re

#pytorch lightning
import pytorch_lightning as ptl

#pytroch
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

#hugginface
from transformers import get_constant_schedule_with_warmup

#am 
from hotam.utils import ensure_numpy, ensure_flat
from hotam.trainer.metrics import Metrics
from hotam import get_logger


logger = get_logger("TRAINING")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def my_mean(scores):

    if isinstance(scores,list):
        scores = np.array(scores)
    
    if scores.shape[0] == 0:
        return scores[0]

    if torch.is_tensor(scores):
        return torch.mean(scores)

    return np.mean(scores)
    

class PTLBase(ptl.LightningModule, Metrics):

    def __init__(   self,  
                    model, 
                    dataset:None, 
                    hyperparamaters:dict,
                    metrics:List[dict],
                    monitor_metric:str, 
                    progress_bar_metrics:list,
                    ):
        super().__init__()
        
        self.dataset = dataset
        self.hyperparamaters = hyperparamaters
        self.model = model(
                            hyperparamaters=hyperparamaters,
                            task2labels={t:ls for t,ls in self.dataset.task2labels.items() if t in self.dataset.main_tasks},
                            feature2dim=self.dataset.feature2dim,
                            )

        self.monitor_metric = monitor_metric
        self.progress_bar_metrics = progress_bar_metrics
        self.metrics, self.metric_names, self.class_metric_names = self.metrics()

                    
    def forward(self) -> dict:
        raise NotImplementedError()


    def log_progress_bar(self, result, metrics):

        #prog_dict = super().get_progress_bar_dict()
        #prog_dict.pop("v_num", None)
        
        for metric in self.scorer.progress_bar_metrics+[self.scorer.monitor_metric]:

            if metric not in metrics:
                continue
            
            result.log(
                        metric, 
                        metrics[metric],
                        on_step=True, 
                        on_epoch=True, 
                        prog_bar=True,
                        reduce_fx=my_mean, 
                        tbptt_reduce_fx=my_mean,
                        logger=False
                        )


    def _step(self, batch, split):
        #self.current_step += 1
        # fetches the device so we can place tensors on the correct memmory
        device = f"cuda:{next(self.parameters()).get_device()}" if self.on_gpu else "cpu"

        self.batch = batch

        #pass on the whole batch to the model
        output_dict = self.model.forward(batch)
    
        if self.logger and split in ["val", "test"]:
            self.logger.log_output(self.reformat_outputs(output_dict), self.current_epoch)
        
        if "total" in output_dict["loss"]:
            total_loss = output_dict["loss"]["total"]
        else:
            total_loss = 0
            for task, loss in output_dict["loss"].items():
                total_loss += loss
            
        metrics = self.score(batch, output_dict, split)

        return  total_loss, metrics
    

    def training_step(self, batch_ids, batch_idx):
        loss, metrics = self._step(batch_ids, "train")
        result = ptl.TrainResult(  
                                    minimize=loss,
                                    #early_stop_on=torch.Tensor([metrics[].get(self.monitor_metric,None)]), 
                                    #checkpoint_on=torch.Tensor([metrics[self.monitor_metric]])       
                                )
        self.log_progress_bar(result, metrics)
        result.log_dict(metrics, on_epoch=True, reduce_fx=my_mean, tbptt_reduce_fx=my_mean)
        return result


    def validation_step(self, batch_ids, batch_idx):
        loss, metrics = self._step(batch_ids, "val")
        result = ptl.EvalResult(
                                    early_stop_on=torch.Tensor([metrics[self.scorer.monitor_metric]]), 
                                    checkpoint_on=torch.Tensor([metrics[self.scorer.monitor_metric]])
                                )
        self.log_progress_bar(result, metrics)
        result.log_dict(metrics, on_epoch=True, reduce_fx=my_mean, tbptt_reduce_fx=my_mean)
        return result


    def test_step(self, batch_ids, batch_idx):
        loss, metrics = self._step(batch_ids, "test")
        result = ptl.EvalResult()
        result.log_dict(metrics, on_epoch=True, reduce_fx=my_mean, tbptt_reduce_fx=my_mean)
        return result
     

    def configure_optimizers(self):

        if self.model.OPT.lower() == "adadelta":
            opt = torch.optim.Adadelta(self.parameters(), lr=self.model.LR)
        elif self.model.OPT.lower() == "sgd":
            opt = torch.optim.SGD(self.parameters(), lr=self.model.LR)
        elif self.model.OPT.lower() == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.model.LR)
        elif self.model.OPT.lower() == "rmsprop":
            opt = torch.optim.RMSprop(self.parameters(), lr=self.model.LR)
        elif self.model.OPT.lower() == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.model.LR)
        else:
            raise KeyError(f'"{self.OPT}" is not a supported optimizer')

        if "scheduler" in self.hyperparamaters:
            if self.hyperparamaters["scheduler"].lower() == "rop":
                scheduler = {
                                'scheduler': ReduceLROnPlateau(opt),
                                'monitor': "val_checkpoint_on",
                                'interval': 'epoch',
                                'frequency': 1
                            }
            elif self.hyperparamaters["scheduler"].lower() == "constant_warmup":
                scheduler = get_constant_schedule_with_warmup(
                                                                optimizer=opt,
                                                                num_warmup_steps=self.hyperparamaters["num_warmup_steps"],
                                                                last_epoch=self.hyperparamaters.get("schedular_last_epoch", -1)

                                                                )
            else:
                raise KeyError(f'"{self.hyperparamaters["scheduler"]} is not a supported learning shedular')

        if "scheduler" in self.hyperparamaters:
            return [opt], [scheduler]
        else:
            return opt


    def reformat_outputs(self,output_dict):

        lengths = ensure_numpy(self.batch["ids"])
        id2idx = {id_:idx for idx, id_ in enumerate(ensure_numpy(self.batch["ids"]))}
        pred_tabel = {str(i):{} for i in id2idx.keys()}
        prob_tabels = {str(i):{} for i in id2idx.keys()}

        get_probs = False
        for task in self.dataset.subtasks:
            labels = self.dataset.task2labels[task] 

            if output_dict["probs"]:
                if task in output_dict["probs"]:
                    get_probs = True

            task_preds = ensure_numpy(output_dict["preds"][task])
            
            for ID, i in id2idx.items():
                pred_tabel[str(ID)][task] = self.dataset.decode_list(task_preds[i][:lengths[i]], task).tolist()

                if get_probs:
                    task_probs = ensure_numpy(output_dict["preds"][task])
                    prob_tabels[str(ID)][task] = {"columns":labels, "data": task_probs[i][:lengths[i]].tolist()}

        return {"preds": pred_tabel, "probs": prob_tabels}


    # def reformat_outputs(self,output_dict):
    #     mask = ensure_numpy(self.batch["mask"])
    #     ids = ensure_numpy(self.batch["id"])
    #     lengths = ensure_numpy(self.batch["id"])
    #     sample_seq_ids = ensure_flat(ensure_numpy([[i]*l for i,l in zip(ids,lengths)]))
    #     pred_tabel = {"sample_id":sample_seq_ids}
    #     prob_tabels = {}

    #     for task in self.dataset.subtasks:
            
    #         task_preds = ensure_flat(ensure_numpy(output_dict["preds"][task]), mask)
    #         preds_str = list(self.dataset.decode_list(task_preds,task))

    #         print( len(sample_seq_ids), len(preds_str))
    #         assert len(sample_seq_ids) == len(preds_str)
    #         pred_tabel[task] = preds_str

    #         if output_dict["probs"]:
    #             if task in output_dict["probs"]:

    #                 task_probs = ensure_flat(ensure_numpy(output_dict["preds"][task]), mask)

    #                 assert len(task_probs) == len(sample_seq_ids)

    #                 labels = self.dataset.task2labels[task]
    #                 prob_tabels[task] = {"columns":labels, "data":task_probs}


    #     return {"preds": pred_tabel, "probs": prob_tabels}


 
