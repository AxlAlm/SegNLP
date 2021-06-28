from segnlp.utils.output import Output
from segnlp.utils.input import Input
import numpy as np
import os
from numpy.lib.arraysetops import isin
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
from segnlp import utils
import segnlp.metrics as metrics

logger = get_logger("PTLBase (ptl.LightningModule)")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class PTLBase(ptl.LightningModule):


    def __init__(   self,  
                    hyperparamaters:dict,
                    tasks:list,
                    all_tasks:list,
                    subtasks:list,
                    label_encoders:dict,
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
        self.tasks = tasks

        if "seg" in subtasks:
            self.seg_task = sorted([task for task in tasks if "seg" in task], key = lambda x: len(x))[0]

        self.metrics = utils.MetricContainer(
                                            metric = metric,
                                            label_encoders = label_encoders,
                                            )

        self.output = utils.Output(
                                        label_encoders = label_encoders, 
                                        tasks = tasks,
                                        all_tasks = all_tasks,
                                        subtasks = subtasks,
                                        prediction_level = prediction_level,
                                        inference = inference,
                                        )

        self.outputs = {"val":[], "test":[], "train":[]}


    def _token_rep(self, batch:utils.Input, output:dict):

        if hasattr(self, "token_rep"):
            stuff = self.token_rep(batch, output)

            assert isinstance(stuff, dict)

            output.add_stuff(stuff)
    

    def _token_clf(self, batch:utils.Input, output:dict):

        if hasattr(self, "token_clf"):
            logits, preds = self.token_clf(batch, output)
            output.add_logits(
                            logits, 
                            task = self.seg_task
                            )
            output.add_preds(
                            preds, 
                            level = "token", 
                            task = self.seg_task
                            )


    def _seg_rep(self, batch:utils.Input, output:dict):

        if hasattr(self, "seg_rep"):
            stuff = self.seg_rep(batch, output)

            assert isinstance(stuff, dict)

            output.add_stuff(stuff)
    

    def _label_clf(self, batch:utils.Input, output:dict):

        if hasattr(self, "label_clf"):
            logits, preds = self.label_clf(batch, output)
            output.add_logits(  logits, 
                                task = "label"
                            )
            output.add_preds(   preds, 
                                level = "seg", 
                                task = "label"
                                )

        
    def _link_clf(self, batch:utils.Input, output:dict):

        if hasattr(self, "link_clf"):
            logits, preds = self.link_clf(batch, output)
            output.add_logits(
                            logits, 
                            task = "link"
                            )
            output.add_preds(
                            preds, 
                            level = "seg", 
                            task = "link"
                            )


    def _link_label_clf(self, batch:utils.Input, output:dict):

        if hasattr(self, "link_label_clf"):
            logits, preds = self.link_label_clf(batch, output)
            output.add_logits(
                            logits, 
                            task = "link_label"
                            )
            output.add_preds(
                            preds, 
                            level = "seg", 
                            task = "link_label"
                            )


    def forward(self, batch:Input, output:Output):

        # 1) represent tokens
        self._token_rep(batch, output)

        # 2) classifiy on tokens
        self._token_clf(batch, output)

        # 3) represent segments
        self._seg_rep(batch, output)

        # 4) classify labels
        self._label_clf(batch, output)

        # 5) classify links
        self._link_clf(batch, output)

        #6) classify link_labels
        self._link_label_clf(batch, output)


    def _step(self, batch:utils.Input, split:str):
        batch.current_epoch = self.current_epoch

        output = self.output.step(batch)

        self.forward(batch, output)

        self.metrics.calc_add(
                            df = output.df, 
                            split = split
                            )
                                

        if self.inference:
            loss = 0
        else:
            loss = self.loss(batch, output)

        return loss, output.df
      
    
    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch, "train")
        self.log('train_loss', loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx): 
        loss, df = self._step(batch, "val")
        self.log('val_loss', loss, prog_bar=True)

        if self.monitor_metric != "val_loss":
            self.log(f'val_{self.monitor_metric}', self.metrics["val"][-1][self.monitor_metric.replace("val_","")], prog_bar=True)

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
     