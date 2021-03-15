
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
from hotam import get_logger
from hotam.nn import ModelInput, ModelOutput


logger = get_logger("PTLBase (ptl.LightningModule)")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



class MetricContainer(dict):

    def __init__(self):
       
        for k in ["train", "val", "test"]:
            self[k] = []


    def add(self, metrics:dict, split:str):
        """
        metrics = {"metic1": 0.3, "metric2": 0.5, ...., "metricn": 0.9 }
        """
        self[split].append(metrics)

        
    def get_epoch_score(self, split:str):
        epoch_metrics = pd.DataFrame(self[split]).mean()
        
        if epoch_metrics.shape[0] == 0:
            return {}

        epoch_metrics.index = split + "_" + epoch_metrics.index
        return epoch_metrics.to_dict()
   



class PTLBase(ptl.LightningModule):


    def __init__(   self,  
                    model:torch.nn.Module, 
                    hyperparamaters:dict,
                    tasks:list,
                    all_tasks:list,
                    label_encoders:dict,
                    prediction_level:str,
                    task_dims:dict,
                    feature_dims:dict,
                    train_mode:bool=True
                    ):
        super().__init__()
        self.hyperparamaters = hyperparamaters
        self.monitor_metric = hyperparamaters.get("monitor_metric", "loss")
        self.prediction_level = prediction_level
        self.tasks = tasks
        self.all_tasks = all_tasks
        self.label_encoders = label_encoders
        self.train_mode = train_mode
        self.model = model(
                            hyperparamaters=hyperparamaters,
                            task_dims=task_dims,
                            feature_dims=feature_dims,
                            train_mode=train_mode
                            )
        self.metrics = MetricContainer()

    def forward(self, batch:ModelInput):
        return self._step(batch, split="test")


    def _step(self, batch:ModelInput, split):
        # fetches the device so we can place tensors on the correct memmory
        #device = f"cuda:{next(self.parameters()).get_device()}" if self.on_gpu else "cpu"

        batch.current_epoch = self.current_epoch

        output = self.model.forward(
                                    batch, 
                                    ModelOutput(
                                            batch=batch,
                                            return_output=True, 
                                            label_encoders=self.label_encoders, 
                                            tasks=self.tasks,
                                            all_tasks=self.all_tasks,
                                            prediction_level=self.prediction_level,
                                            calc_metrics=True if self.train_mode else False, 
                                            )
                                    )
        # metrics = {f"{split}_"+k:v for k,v in output.metrics.items()}
        # metrics["epoch"] = self.current_epoch
        #print(metrics)
        # self.logger.experiment.log_metric(f"{split}_f1", metrics.get(f"{split}_f1",0))
        # self.logger.log_metrics(metrics)

        self.metrics.add(output.metrics, split)
        return output.loss.get("total", 0), output
      
    

    def training_step(self, batch, batch_idx):
        loss, output = self._step(batch, "train")
        self.log('train_loss', loss, prog_bar=True)

        # if self.monitor_metric != "loss":
        #     self.log(f'train_{self.monitor_metric}', self.metrics["train"][self.monitor_metric], prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx): 
        loss, output = self._step(batch, "val")
        self.log('val_loss', loss, prog_bar=True)

        if self.monitor_metric != "val_loss":
            self.log(f'val_{self.monitor_metric}', self.metrics["val"][-1][self.monitor_metric.replace("val","")], prog_bar=True)

        return {"val_loss": loss}


    def test_step(self, batch, batch_idx):
        _, output = self._step(batch, "test")
        return output



    # def training_step_end(self, outs):
    #     return outs


    # def validation_step_end(self, outs):
    #     return outs


    # def test_step_end(self, outs):
    #     return outs
     

    def _end_of_epoch(self, split):
        #if self.logger is not None:
        epoch_metrics = self.metrics.get_epoch_score(split)

        # if isinstance(self.logger, CometLogger):
        #     for m in epoch_metrics:
        #         if "confusion_matrix" in m:
        #             self.logger.experiment.log_confusion_matrix(labels=["one", "two", "three"],
        #                                                         matrix=[[10, 0, 0],
        #                                                                 [ 0, 9, 1],
        #                                                                 [ 1, 1, 8]])

        print("LOGGIN DICT", split, epoch_metrics)
        self.log_dict(
                        epoch_metrics,
                        on_step=False,
                        on_epoch=True,
                        )
        

    def on_train_epoch_end(self, *args, **kwargs):
        self._end_of_epoch("train")


    def on_validation_epoch_end(self):
        self._end_of_epoch("val")
        self.metrics = MetricContainer()


    def on_test_epoch_end(self):
        self._end_of_epoch("test")



    # def validation_step(self, batch_ids, batch_idx):
    #     loss, metrics = self._step(batch_ids, "val")
    #     # result = ptl.EvalResult(
    #     #                             early_stop_on=torch.Tensor([metrics[self.monitor_metric]]), 
    #     #                             checkpoint_on=torch.Tensor([metrics[self.monitor_metric]])
    #     #                         )
    #     # result.log_dict(metrics, on_epoch=True, reduce_fx=my_mean, tbptt_reduce_fx=my_mean)
    #     #return result


    # def test_step(self, batch_ids, batch_idx):
    #     loss, metrics = self._step(batch_ids, "test")
    #     result = ptl.EvalResult()
    #     result.log_dict(metrics, on_epoch=True, reduce_fx=my_mean, tbptt_reduce_fx=my_mean)
    #     #return result
     

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
     








