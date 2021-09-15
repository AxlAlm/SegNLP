

from torch.nn.modules import module
from segnlp.layer_wrappers.layer_wrappers import Layer
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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

#hugginface
from transformers import get_constant_schedule_with_warmup

#segnlp
from segnlp import get_logger
from segnlp import utils
from segnlp.utils import LabelEncoder
from segnlp.layer_wrappers.layer_wrappers import Layer
from segnlp.layer_wrappers import TokenEmbedder
from segnlp.layer_wrappers import SegEmbedder
from segnlp.layer_wrappers import Encoder
from segnlp.layer_wrappers import Segmenter
from segnlp.layer_wrappers import PairRep
from segnlp.layer_wrappers import SegRep
from segnlp.layer_wrappers import Labeler
from segnlp.layer_wrappers import LinkLabeler
from segnlp.layer_wrappers import Linker


logger = get_logger("PTLBase (ptl.LightningModule)")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class PTLBase(ptl.LightningModule):


    def __init__(   self,  
                    hyperparamaters:dict,
                    label_encoder: LabelEncoder,
                    feature_dims:dict,
                    metric:Union[Callable,str],
                    logger=None,
                    inference:bool=False
                    ):
        super().__init__()
        self.hps = hyperparamaters
        self.monitor_metric = hyperparamaters["general"].get("monitor_metric", "loss")
        self.feature_dims = feature_dims
        self.task_dims = {task: len(labels) for task, labels in label_encoder.task_labels.items()}
        self.inference = inference
        self.seg_task = label_encoder.seg_task
        self.logger = logger

        # setting up metric container which takes care of metric calculation, aggregation and storing
        self.metrics = utils.MetricContainer(
                                            metric = metric,
                                            task_labels = label_encoder.task_labels,
                                            )

        # setting up an output object which will format and store outputs for each batch.
        # using .step() will create a batch specific output container which will be used to store information 
        # throughout the step
        self.output = utils.BatchOutput(
                                        label_encoder = label_encoder, 
                                        seg_gts_k = self.hps["general"].get("seg_gts_k", None),
                                        )

        # the batch outputs can be collected and stored to get outputs over a complete split. E.g. returning 
        # test outputs or validation outputs
        self.outputs = {"val":[], "test":[], "train":[]}

        # we will add modules that we will freeze
        self.freeze_modules = set()

        # we will add modules that we will skip
        self.skip_modules = set()

        # if one wants to train a part of a network first one can skip a section of the model
        # skipping, unlike freezing, will skip all calls to the sections you decide to skip
        # Skipping will also freeze layer weights
        if hyperparamaters["general"].get("skip_token_module", False):
            self.skip_modules.add("token_module")
            self.freeze_modules.add("token_module")

        if hyperparamaters["general"].get("skip_segment_module", False):
            self.skip_modules.add("segment_module")
            self.freeze_modules.add("segment_module")

        # Freezing will freeze the weights in all layers in the section
        if hyperparamaters["general"].get("freeze_token_module", False):
            self.freeze_modules.add("token_module")

        if hyperparamaters["general"].get("freeze_segment_module", False):
            self.freeze_modules.add("segment_module")

        
    @classmethod
    def name(self):
        return self.__name__


    def __rep(self, batch:utils.BatchInput, output:utils.BatchOutput, f_name:str):
        
        if not hasattr(self,  f_name):
            return 

        stuff = getattr(self, f_name)(batch, output)

        assert isinstance(stuff, dict)
        output.add_stuff(stuff)


    def __clf(self, batch:utils.BatchInput, output:utils.BatchOutput, f_name:str):
        
        if not hasattr(self,  f_name):
            return 

        task_outs = getattr(self, f_name)(batch, output)

        assert isinstance(task_outs, list)

        for task_dict in task_outs:

            if "task" in task_dict:
                output.add_preds(
                                task_dict["preds"], 
                                level = task_dict["level"],
                                task = task_dict["task"]
                                )

            if "logits" in task_dict:
                output.add_logits(
                                    task_dict["logits"], 
                                    task = task_dict["task"]
                                    )


    def __token_rep(self, batch:utils.BatchInput, output:utils.BatchOutput):
        self.__rep(
                    batch=batch, 
                    output=output, 
                    f_name = "token_rep"
                    )
    

    def __token_clf(self, batch:utils.BatchInput, output:utils.BatchOutput):
        self.__clf(
                    batch = batch, 
                    output = output, 
                    f_name = "token_clf"
                    )
        

    def __seg_rep(self, batch:utils.BatchInput, output:utils.BatchOutput):
        self.__rep(
                    batch=batch, 
                    output=output, 
                    f_name = "seg_rep"
                    )
    

    def __seg_clf(self, batch:utils.BatchInput, output:utils.BatchOutput):
        self.__clf(
                    batch=batch, 
                    output=output, 
                    f_name = "seg_clf"
                    )


    def forward(self, batch:utils.BatchInput, output:utils.BatchOutput):

        ## will run every module and add stuff to output.

        # we skip the token_module
        if "token_module" not in self.skip_modules:
            
            # 1) represent tokens
            self.__token_rep(batch, output)

            # 2) classifiy on tokens
            self.__token_clf(batch, output)

        # we freeze the segment module
        if "segment_module" not in self.skip_modules:

            # 3) represent segments
            self.__seg_rep(batch, output)

            # 4) classify segments
            self.__seg_clf(batch, output)


    def _step(self, batch:utils.BatchInput, split:str):
        batch.current_epoch = self.current_epoch

        # creates a batch specific output container which will be filled
        # with predictions, logits and outputs of modules and submodules
        output = self.output.step(batch, step_type = split)


        # pass the batch and output through the modules
        self.forward(batch, output)

        # Will take the prediction dataframe created during the forward pass
        # and pass it to the metric container which will calculate, aggregate
        # and store metrics
        self.metrics.calc_add(
                            df = output.df.copy(deep=True), 
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
            score = self.metrics["val"][-1][self.monitor_metric.replace("val_","")]
            self.log(self.monitor_metric, score, prog_bar=True)

        return {"val_loss": loss}


    def test_step(self, batch, batch_idx):
        loss, df = self._step(batch, "test")
        self.outputs["test"].extend(df.to_dict("records"))
        return loss


    def on_train_epoch_end(self):
        self._end_of_epoch("train")


    def on_validation_epoch_end(self):
        self._end_of_epoch("val")


    def on_test_epoch_end(self):
        self._end_of_epoch("test")


    def _end_of_epoch(self, split):
        epoch_metrics = self.metrics.calc_epoch_metrics(split)

        logger.log_epoch(
                        random_seed = "RANDOM SEED",
                        hyperparamater_id =  "SOMETHING",
                        epoch_metrics = epoch_metrics,
                        )
                        
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


    def __add_layer(self, layer:Layer, args:tuple, kwargs:dict):

        module = kwargs.pop("module")

        assert module in ["token_module", "segment_module"], f'"{module}" is not a supported module. Chose on of {["token_module", "segment_module"]}'

        if module in self.freeze_modules:

            #if isinstance(layer, Layer):
            kwargs["hyperparamaters"]["freeze"] = True

            # elif isinstance(layer, nn.Module):
            #     utils.freeze_model(layer)

        return layer(*args, **kwargs)


    def add_token_embedder(self, *args, **kwargs):
        kwargs["module"] = "token_module"
        return self.__add_layer(TokenEmbedder, args=args, kwargs=kwargs)


    def add_seg_embedder(self, *args, **kwargs):
        kwargs["module"] = "segment_module"
        return self.__add_layer(SegEmbedder, args=args, kwargs=kwargs)


    def add_encoder(self, *args, **kwargs):
        return self.__add_layer(Encoder, args=args, kwargs=kwargs)


    def add_pair_rep(self, *args, **kwargs):
        kwargs["module"] = "segment_module"
        return self.__add_layer(PairRep, args=args, kwargs=kwargs)


    def add_seg_rep(self, *args, **kwargs):
        kwargs["module"] = "segment_module"
        return self.__add_layer(SegRep, args=args, kwargs=kwargs)


    def add_segmenter(self, *args, **kwargs):
        kwargs["task"] = self.seg_task
        kwargs["module"] = "token_module"
        return self.__add_layer(Segmenter, 
                                args=args,
                                kwargs=kwargs,
                                )


    def add_labeler(self, *args, **kwargs):
        kwargs["module"] = "segment_module"
        return self.__add_layer(
                                Labeler,                                 
                                args=args,
                                kwargs=kwargs,
                                )


    def add_link_labeler(self, *args, **kwargs):
        kwargs["module"] = "segment_module"
        return self.__add_layer(
                                LinkLabeler,                                 
                                args=args,
                                kwargs=kwargs,
                                )


    def add_linker(self, *args, **kwargs):
        kwargs["module"] = "segment_module"
        return self.__add_layer(
                                Linker,                                 
                                args=args,
                                kwargs=kwargs,
                                )
