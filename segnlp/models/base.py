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
from segnlp import get_logger
import segnlp.utils as utils
import segnlp.metrics as metrics

logger = get_logger("PTLBase (ptl.LightningModule)")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class PTLBase(ptl.LightningModule):


    def __init__(   self,  
                    hyperparamaters:dict,
                    tasks:list,
                    all_tasks:list,
                    label_encoders:dict,
                    task_labels:dict,
                    prediction_level:str,
                    task_dims:dict,
                    feature_dims:dict,
                    metric:Union[Callable,str],
                    inference:bool=False
                    ):
        super().__init__()
        self.hps = hyperparamaters
        self.monitor_metric = hyperparamaters["general"].get("monitor_metric", "loss")
        self.feature_dims = feature_dims
        self.task_dims = task_dims
        self.inference = inference
        self.task_labels = task_labels
        self.tasks = tasks

        self.metrics = utils.MetricContainer(
                                            metric = metric
                                            )

        self.formater = utils.OutputFormater(
                                        label_encoders=label_encoders, 
                                        tasks=tasks,
                                        all_tasks=all_tasks,
                                        prediction_level=prediction_level,
                                        inference = inference,
                                        )

        self.outputs = {"val":[], "test":[], "train":[]}


    def forward(self, batch:utils.ModelInput):
        return self._step(batch, split="test")


    def _step(self, batch:utils.ModelInput, split:str):
        batch.current_epoch = self.current_epoch
        output = self.forward(batch)
        df = self.formater.format(
                                    batch=batch,
                                    preds=output["preds"],
                                    )
        self.metrics.calc_add(
                            df=df, 
                            task_labels=self.task_labels,
                            split=split
                            )
                                

        if self.inference:
            loss = 0
        else:
            loss = self.loss(batch, output)

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

        #self.outputs["val"].extend(df.to_dict("records"))
        return {"val_loss": loss}


    def test_step(self, batch, batch_idx):
        loss, df = self._step(batch, "test")
        self.outputs["test"].extend(df.to_dict("records"))
        return loss


    def on_train_epoch_end(self, *args, **kwargs):
        self._end_of_epoch("train")


    def on_validation_epoch_end(self):
        self._end_of_epoch("val")


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

        opt_class = getattr(torch.optim, self.hps["general"]["optimizer"])
        opt = opt_class(self.parameters(), lr = self.hps["general"]["lr"])


        if "scheduler" in self.hps["general"]:
            if self.hps["general"]["scheduler"].lower() == "rop":
                scheduler = {
                                'scheduler': ReduceLROnPlateau(
                                                                opt,
                                                                patience=5,
                                                                factor=0.001
                                                                ),
                                'monitor': "val_loss",
                                'interval': 'epoch',
                            }
            elif self.hps["general"]["scheduler"].lower() == "constant_warmup":
                scheduler = get_constant_schedule_with_warmup(
                                                                optimizer=opt,
                                                                num_warmup_steps=self.hps["general"]["num_warmup_steps"],
                                                                last_epoch=self.hps["general"].get("schedular_last_epoch", -1)

                                                                )
            else:
                raise KeyError(f'"{self.hps["general"]["scheduler"]} is not a supported learning shedular')

        if "scheduler" in self.hps["general"]:
            return [opt], [scheduler]
        else:
            return opt
     