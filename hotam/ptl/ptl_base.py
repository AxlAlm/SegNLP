
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


def my_mean(scores):

    scores = ensure_numpy(scores)
    
    if scores.shape[0] == 0:
        return scores[0]

    if torch.is_tensor(scores):
        return torch.mean(scores)
    
    if len(scores.shape) > 1:
        return np.mean(scores, axis=0)
    else:
        return np.mean(scores)
    

class PTLBase(ptl.LightningModule):


    def __init__(   self,  
                    model:torch.nn.Module, 
                    hyperparamaters:dict,
                    all_tasks:list,
                    label_encoders:dict,
                    prediction_level:str,
                    task_dims:dict,
                    feature_dims:dict,
                    ):
        super().__init__()
        self.hyperparamaters = hyperparamaters
        self.monitor_metric = hyperparamaters.get("monitor_metric", "loss")
        self.prediction_level = prediction_level
        self.all_tasks = all_tasks
        self.label_encoders = label_encoders
        self.model = model(
                            hyperparamaters=hyperparamaters,
                            task_dims=task_dims,
                            feature_dims=feature_dims,
                            )
                    

    def forward(self) -> dict:
        raise NotImplementedError()


    def _step(self, batch:ModelInput, split):

        # fetches the device so we can place tensors on the correct memmory
        device = f"cuda:{next(self.parameters()).get_device()}" if self.on_gpu else "cpu"
        self.split = split 
        batch.current_epoch = self.current_epoch

        output = self.model.forward(
                                    batch, 
                                    ModelOutput(
                                            batch=batch,
                                            calc_metrics=True, 
                                            return_output=True, 
                                            label_encoders=self.label_encoders, 
                                            all_tasks=self.all_tasks,
                                            prediction_level=self.prediction_level,
                                            )
                                    )

        # get the metrics and the loss (gpu grad loss)
        metrics = output.metrics
        total_loss = output.loss["total"]

        # if we are on val and test we can log our output
        if self.logger and split in ["val", "test"]:
            self.logger.log_outputs(output.outputs)
            
        return  total_loss, metrics
    

    def training_step(self, batch_ids, batch_idx):
        loss, metrics = self._step(batch_ids, "train")
        result = ptl.TrainResult(  
                                    minimize=loss,

                                )
        result.log_dict(metrics, on_epoch=True, reduce_fx=my_mean, tbptt_reduce_fx=my_mean)
        return result


    def validation_step(self, batch_ids, batch_idx):
        loss, metrics = self._step(batch_ids, "val")
        result = ptl.EvalResult(
                                    early_stop_on=torch.Tensor([metrics[self.monitor_metric]]), 
                                    checkpoint_on=torch.Tensor([metrics[self.monitor_metric]])
                                )
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
     








