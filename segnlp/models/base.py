import numpy as np
import os
import pandas as pd
from typing import List, Dict, Union, Tuple, Callable
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
from segnlp.utils import ensure_numpy
from segnlp.utils import ensure_flat
from segnlp import get_logger
from segnlp.nn.utils import ModelInput
from segnlp.nn.utils import ModelOutput

logger = get_logger("PTLBase (ptl.LightningModule)")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from segnlp.utils import timer

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
                    inference:bool=False
                    ):
        super().__init__()
        self.hyperparamaters = hyperparamaters
        self.monitor_metric = hyperparamaters.get("monitor_metric", "loss")
        
        self.model = model(
                            hyperparamaters=hyperparamaters,
                            task_dims=task_dims,
                            feature_dims=feature_dims,
                            inference=inference
                            )

        self.metrics = MetricContainer(
                                        metric_fn = metric_fn
                                        )

        self.formater = OutputFormater(
     
                                    label_encoders=label_encoders, 
                                    tasks=tasks,
                                    all_tasks=all_tasks,
                                    prediction_level=prediction_level,
                                    inference = inference,
                                    )

        self.outputs = {"val":[], "test":[], "train":[]}


    def forward(self, batch:ModelInput):
        return self._step(batch, split="test")


    def _step(self, batch:ModelInput, split):
        batch.current_epoch = self.current_epoch
        loss, preds = self.model.forward(batch)
        df = self.formater.format(
                                    input=batch,
                                    output=preds,
                                    )
        
        self.metrics.calc_add(df, split)
        return loss, df
      
    
    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch, "train")
        self.log('train_loss', loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx): 
        loss, df = self._step(batch, "val")
        self.log('val_loss', loss, prog_bar=True)

        if self.monitor_metric != "val_loss":
            self.log(f'val_{self.monitor_metric}', self.metrics["val"][-1][self.monitor_metric.replace("val_","")], prog_bar=True)

        self.outputs["val"].extend(df.to_dict("records"))
        return {"val_loss": loss}


    def test_step(self, batch, batch_idx):
        _, output = self._step(batch, "test")
        self.outputs["test"].extend(df.to_dict("records"))
        return output


    def on_train_epoch_end(self, *args, **kwargs):
        self._end_of_epoch("train")


    def on_validation_epoch_end(self):
        self._end_of_epoch("val")
        self.metrics = MetricContainer()


    def on_test_epoch_end(self):
        self._end_of_epoch("test")


    def _end_of_epoch(self, split):
        epoch_metrics = self.metrics.calc_epoch_metrics(split)
        self.log_dict(
                        epoch_metrics,
                        on_step=False,
                        on_epoch=True,
                        )


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
                                'scheduler': ReduceLROnPlateau(
                                                                opt,
                                                                patience=5,
                                                                factor=0.001
                                                                ),
                                'monitor': "val_loss",
                                'interval': 'epoch',
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
     