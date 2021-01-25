
#basics
import numpy as np
import warnings
import os
import pandas as pd
from typing import List, Dict, Union, Tuple
import re
from copy import deepcopy

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
    
    if len(scores.shape) > 1:
        return np.mean(scores, axis=0)
    else:
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
        
        for metric in self.progress_bar_metrics+[self.monitor_metric]:

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
        self.batch.current_epoch = self.current_epoch

        #pass on the whole batch to the model
        output_dict = self.model.forward(self.batch)

        for task, preds in output_dict["preds"].items():
            assert torch.is_tensor(preds), f"{task} preds need to be a tensor"
            assert len(preds.shape) == 2, f"{task} preds need to be a 2D tensor"

        for task, probs in output_dict["probs"].items():
            assert torch.is_tensor(probs), f"{task} preds need to be a tensor"
            assert len(probs.shape) == 3, f"{task} preds need to be a 2D tensor"


        if "total" in output_dict["loss"]:
            total_loss = output_dict["loss"]["total"]
        else:
            total_loss = 0
            for task, loss in output_dict["loss"].items():
                total_loss += loss
        
        metrics = self.score(self.batch, output_dict, split)
    
        if self.logger and split in ["val", "test"]:
            self.logger.log_outputs(self.__reformat_outputs(output_dict))
        
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
                                    early_stop_on=torch.Tensor([metrics[self.monitor_metric]]), 
                                    checkpoint_on=torch.Tensor([metrics[self.monitor_metric]])
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

    
    def __BIO_decode(self, bio_labels):
        """
        
        """
        #bio_labels_str = "-".join(bio_labels.astype(str))
        bio_labels_str = "-".join(bio_labels)
        self.__span_id = 0
        def repl(m):
            c = f'SPAN_{self.__span_id}-' * len(m.group())
            self.__span_id += 1
            return c
            
        m = re.sub(fr"B-(I(-|))*", repl, bio_labels_str)
        return m.split("-")
            

    def __reformat_outputs(self, output_dict):

        ids_to_log = np.array(self.dataset.config["tracked_sample_ids"]["0"])

        length_type = "lengths_seq" if self.batch.prediction_level == "ac" else "lengths_tok"
        id2idx = {  
                    str(ID):(i, length) for i, (ID, length) in 
                    enumerate(zip(ensure_numpy(self.batch["ids"]), ensure_numpy(self.batch[length_type])))
                    if ID in ids_to_log
                    }

        if ids_to_log.shape[0] == 0:
            return {}

        outputs = {ID:{"preds":{}, "probs":{}, "gold":{}, "text":{}} for ID in id2idx.keys()}

        for ID, (i, length) in id2idx.items():
            
            #NOTE! this should not be needed.. fix the origin problem thx
            if self.batch.prediction_level == "ac":
                outputs[ID]["text"] = [t.tolist() if isinstance(t, np.ndarray) else t for t in self.batch["text"][i].tolist()]
            else:
                outputs[ID]["text"] = self.batch["text"][i].tolist()


            spans_added = False

            for task in self.dataset.subtasks:
                task_preds = ensure_numpy(output_dict["preds"][task])
                task_gold = ensure_numpy(self.batch[task])
          
                outputs[ID]["preds"][task] = self.dataset.decode_list(task_preds[i][:length], task).tolist()
                outputs[ID]["gold"][task] = self.dataset.decode_list(task_gold[i][:length], task).tolist()

                if task == "seg" and not spans_added:
                    outputs[ID]["preds"]["span_ids"] = self.__BIO_decode(outputs[ID]["preds"][task])
                    outputs[ID]["gold"]["span_ids"] = self.__BIO_decode(outputs[ID]["gold"][task])
                    spans_added = True

                if not output_dict["probs"]:
                    continue

                if task not in output_dict["probs"]:
                    continue

                task_probs = ensure_numpy(output_dict["preds"][task])
                outputs[ID]["probs"][task] = {"columns":self.dataset.task2labels[task], "data": task_probs[i][:length].tolist()}
            

        return outputs


 
