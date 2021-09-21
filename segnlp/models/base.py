
#basics
import segnlp
import numpy as np
import os
from typing import List, Dict, Union, Tuple, Callable


#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from segnlp import get_logger
from segnlp import utils
from segnlp.utils import LabelEncoder

from segnlp.layers import link_labelers
from segnlp.layers import linkers
from segnlp.layers import segmenters
from segnlp.layers import seg_embedders
from segnlp.layers import token_embedders
from segnlp.layers import encoders
from segnlp.layers import pair_reps
from segnlp.layers import seg_reps
from segnlp.layers import general


logger = get_logger("BaseModel")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class BaseModel(nn.Module):


    def __init__(   self,  
                    hyperparamaters:dict,
                    label_encoder: LabelEncoder,
                    feature_dims:dict,
                    metric:Union[Callable,str],
                    inference:bool=False
                    ):
        super().__init__()
        self.hps : dict = hyperparamaters
        self.feature_dims : Dict[str, int] = feature_dims
        self.task_dims : Dict[str, int] = {task: len(labels) for task, labels in label_encoder.task_labels.items()}
        self.inference : bool = inference
        self.seg_task : dict = label_encoder.seg_task

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
                                        )

        # the batch outputs can be collected and stored to get outputs over a complete split. E.g. returning 
        # test outputs or validation outputs
        self.outputs : Dict[str, dict[str, float]] = {"val":[], "test":[], "train":[]}


        # for all layers that do classification we need to know which task they are classifying 
        # so we keep track on this by creating a mapping dict. Things will be added to this
        # during self.add_XXX() calls in model.__init__()
        self._layer2task : Dict[str, List[str]]  = {}

        #save all nn.Modules that belong in the token module
        self._token_layers : List[nn.Module] = []

        #save all nn.Modules that belong in the segment module
        self._segment_layers : List[nn.Module]  = []

        # we will add modules that we will freeze
        self._token_layers_are_frozen : bool = False

        # we will add modules that we will skip
        self._segment_layers_are_frozen : bool = False

        
    @classmethod
    def name(self):
        return self.__name__


    @utils.timer
    def __rep(self, batch:utils.BatchInput, output:utils.BatchOutput, module_type:str):
        
        f_name = f"{module_type}_rep"

        if not hasattr(self,  f_name):
            return 

        stuff = getattr(self, f_name)(batch, output)

        assert isinstance(stuff, dict)
        output.add_stuff(stuff)

    @utils.timer
    def __clf(self, batch:utils.BatchInput, output:utils.BatchOutput, module_type:str):
        
        f_name = f"{module_type}_clf"

        level = "seg" if module_type == "segment" else "token"

        if not hasattr(self,  f_name):
            return 

        task_outs = getattr(self, f_name)(batch, output)

        assert isinstance(task_outs, list)

        for task_dict in task_outs:

            level = level if "level" not in task_dict else task_dict["level"]

            if "task" in task_dict:
                output.add_preds(
                                task_dict["preds"], 
                                level = level,
                                task = task_dict["task"]
                                )

            if "logits" in task_dict:
                output.add_logits(
                                    task_dict["logits"], 
                                    task = task_dict["task"]
                                    )

    @utils.timer
    def forward(self, 
                batch:utils.BatchInput, 
                split:str, 
                use_target_segs : bool = False
                ):

        # creates a batch specific output container which will be filled
        # with predictions, logits and outputs of modules and submodules
        output = self.output.step( 
                                    batch, 
                                    use_target_segs = use_target_segs
                                    )

        total_loss = 0

        # we skip the token_module
        if  not self._token_layers_are_frozen:
            
            # 1) represent tokens
            self.__rep(batch, output, "token")

            # 2) classifiy on tokens
            self.__clf(batch, output, "token")


            if not self.inference:
                total_loss += self.token_loss(batch, output)


        # we freeze the segment module
        if not self._segment_layers_are_frozen:

            # 3) represent segments
            self.__rep(batch, output, "seg")

            # 4) classify segments
            self.__clf(batch, output, "seg")


            if not self.inference:
                total_loss += self.seg_loss(batch, output)


        # Will take the prediction dataframe created during the forward pass
        # and pass it to the metric container which will calculate, aggregate
        # and store metrics
        self.metrics.calc_add(
                            df = output.df.copy(deep=True), 
                            split = split
                            )

        return total_loss


    #### LAYER ADDING
    def __add_layer(self, 
                    layer : str, 
                    hyperparamaters : dict,
                    layer_modules : List,
                    module_type: str,
                    input_size : int = None,
                    output_size : int = None,
                    task : str = None
                    ):

        if input_size:
            hyperparamaters["input_size"] = input_size

        if output_size:
            hyperparamaters["output_size"] = output_size

        if task:
            self._layer2task[layer] = task

        for lm in layer_modules:

            try:
                layer = getattr(lm, layer)
            except AttributeError:
                continue
        
        layer = layer(**hyperparamaters)

        if module_type == "token":
            self._token_layers.append(layer)
        else:
            self._segment_layers.append(layer)

        return layer


    def add_token_embedder(  self, 
                        layer: str,  
                        hyperparamaters: dict,
                        ):
        return self.__add_layer(
                                    layer = layer,
                                    hyperparamaters = hyperparamaters,
                                    layer_modules = [token_embedders],
                                    module_type = "token",
                                    )


    def add_seg_embedder(  self, 
                        layer: str,  
                        hyperparamaters: dict,
                        ):
        return self.__add_layer(
                                    layer = layer,
                                    hyperparamaters = hyperparamaters,
                                    layer_modules = [seg_embedders],
                                    module_type = "segment",
                                    )


    def add_token_encoder(  self, 
                        layer: str,  
                        hyperparamaters: dict,
                        input_size : int,
                        ):
        return self.__add_layer(
                                    layer = layer,
                                    hyperparamaters = hyperparamaters,
                                    layer_modules = [encoders],
                                    module_type = "token",
                                    input_size = input_size,
                                    )


    def add_seg_encoder(  self, 
                        layer: str,  
                        hyperparamaters: dict,
                        input_size : int,
                        ):
        return self.__add_layer(
                                    layer = layer,
                                    hyperparamaters = hyperparamaters,
                                    layer_modules = [encoders],
                                    module_type = "segment",
                                    input_size = input_size,
                                    )


    def add_pair_rep(  self, 
                        layer: str,  
                        hyperparamaters: dict,
                        input_size : int,
                        ):
        return self.__add_layer(
                                    layer = layer,
                                    hyperparamaters = hyperparamaters,
                                    layer_modules = [pair_reps],
                                    module_type = "segment",
                                    input_size = input_size,
                                    )


    def add_seg_rep(  self, 
                        layer: str,  
                        hyperparamaters: dict,
                        input_size : int,
                        ):
        return self.__add_layer(
                                    layer = layer,
                                    hyperparamaters = hyperparamaters,
                                    layer_modules = [seg_reps],
                                    module_type = "segment",
                                    input_size = input_size,
                                    )


    def add_segmenter(  self, 
                        layer: str,  
                        hyperparamaters: dict,
                        input_size : int,
                        output_size :int
                        ):
        return self.__add_layer(
                                layer = layer,
                                hyperparamaters = hyperparamaters,
                                layer_modules = [general, segmenters],
                                module_type = "segment",
                                input_size  = input_size,
                                output_size =output_size,
                                task = self.seg_task,
                                )


    def add_labeler(self, 
                        layer: str,  
                        hyperparamaters: dict,
                        input_size : int,
                        output_size : int,
                        ):
        return self.__add_layer(
                                layer = layer,
                                hyperparamaters = hyperparamaters,
                                layer_modules = [general, Labeler],
                                module_type = "segment",
                                input_size = input_size,
                                output_size  = output_size,
                                task = "label",
                                )


    def add_link_labeler(self, 
                        layer: str,  
                        hyperparamaters: dict,
                        input_size : int,
                        output_size : int,
                        ):

        if layer == "DirLinkLabelers":
            task = ["link", "link_label"]
        else:
            task = ["link_label"]

        return self.__add_layer(
                                layer = layer,
                                hyperparamaters = hyperparamaters,
                                layer_modules = [general, link_labelers],
                                module_type = "segment",
                                input_size = input_size,
                                output_size  = output_size,
                                task = task,
                                )


    def add_linker(self, 
                    layer: str,  
                    hyperparamaters: dict,
                    input_size : int,
                    output_size : int,
                    ):
        return self.__add_layer(
                            layer = layer,
                            hyperparamaters = hyperparamaters,
                            layer_modules = [general, linkers],
                            module_type = "segment",
                            input_size = input_size,
                            output_size  = output_size,
                            task = "link",
                            )


    def train(self, freeze_token_module: bool = False, freeze_segment_module: bool = False):
        super().train()

        #make sure to reset them on each epoch call
        self._token_layers_are_frozen  = False
        self._segment_layers_are_frozen = False


        if freeze_token_module:
            self._token_layers_are_frozen =  True
            
            for submodule in self._token_layers:
                utils.freeze_module(submodule)


        if freeze_segment_module:
            self._segment_layers_are_frozen =  True

            for submodule in self._segment_layers:
                utils.freeze_module(submodule)


    def eval(self):
        super().eval()

        #make sure to reset them on each epoch call
        self._token_layers_are_frozen  = False
        self._segment_layers_are_frozen = False



