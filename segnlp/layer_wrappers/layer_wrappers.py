#basics
import inspect
from typing import Union

#pytorch
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from segnlp.layers import encoders 
from segnlp.layers import token_embedders
from segnlp.layers import seg_embedders
from segnlp.layers import seg_reps
from segnlp.layers import pair_reps
from segnlp.layers import linkers
from segnlp.layers import segmenters
from segnlp.layers import general
from segnlp.layers import link_labelers


class Layer(nn.Module):

    def __init__(self, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int=None, 
                output_size:int=None,
                ):
        super().__init__()
        input_size = input_size

        params = hyperparams
        params["input_size"] = input_size
        params["output_size"] = output_size

        params = self.__filter_paramaters(layer, params)

        self.layer = layer(**params)

        if "pretrain_weights" in params:
            self.__load_weights(params["pretrained_weights"])

        if hasattr(self.layer, "output_size"):
            self.output_size = self.layer.output_size
        
        if params.get("freeze", False):
            self.__freeze()


    def __load_weights(self, path_to_weights:str):
        self.layer.load_state_dict(torch.load(path_to_weights))


    def __freeze(self):
        for param in self.layer.parameters():
            param.requires_grad = False


    def __filter_paramaters(self, layer, params):
        sig = inspect.signature(layer)

        if "args" in sig.parameters  or "kwargs" in sig.parameters:
            return params

        filtered_params = {}
        for para in sig.parameters:

            if para in params:
                filtered_params[para] = params[para]
        
        return filtered_params


    def forward(self, *args, **kwargs):
        #kwargs = self.__filter_paramaters(**kwargs)
        return self.layer(*args, **kwargs)


class SegEmbedder(Layer):

    def __init__(self, layer:nn.Module, hyperparams:dict):
        
        if isinstance(layer, str):
            layer = getattr(seg_embedders, layer)

        super().__init__(layer=layer, hyperparams=hyperparams)
        

class TokenEmbedder(Layer):

    def __init__(self, layer:nn.Module, hyperparams:dict):
        
        if isinstance(layer, str):
            layer = getattr(token_embedders, layer)

        super().__init__(layer=layer, hyperparams=hyperparams)
        

class Encoder(Layer):

    def __init__(self, layer:nn.Module, hyperparams:dict, input_size:int):
        
        if isinstance(layer, str):
            layer = getattr(encoders, layer)

        super().__init__(layer=layer, hyperparams=hyperparams, input_size=input_size)
        

class SegRep(Layer):
    
    def __init__(self, layer:nn.Module, hyperparams:dict, input_size:int):
        
        if isinstance(layer, str):
            layer = getattr(seg_reps, layer)

        super().__init__(layer=layer, hyperparams=hyperparams, input_size=input_size)


class PairRep(Layer):
    
    def __init__(self, layer:nn.Module, hyperparams:dict, input_size:int):
        
        if isinstance(layer, str):
            layer = getattr(pair_reps, layer)

        super().__init__(layer=layer, hyperparams=hyperparams, input_size=input_size)


class CLFlayer(Layer):
    """
    Layer which works on segment level
    """

    def __init__(self, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                ):
        super().__init__(              
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )


    def forward(self, *args, **kwargs):
        logits, preds = self.layer(*args, **kwargs)

        return [{
                "task": self.task,
                "level": self.level,
                "logits": logits,
                "preds": preds,
                }]


    def loss(self, *args, **kwargs):
        return self.layer.loss(*args, **kwargs)


class Segmenter(CLFlayer):
    """
    Layer which works on token level
    """

    def __init__(self, 
                layer:Union[nn.Module, str], 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                task:str,
                ):
        self.task = task
        self.level = "token"

        if isinstance(layer, str):
            layer = getattr(segmenters, layer)


        super().__init__(
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )


class Labeler(CLFlayer):
    """
    Layer which works on segment level
    """

    def __init__(self, 
                layer:Union[nn.Module, str],
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                ):
        self.task = "label"
        self.level = "seg"

        if isinstance(layer, str):
            try:
                layer = getattr(general, layer)
            except:
                layer = getattr(Labeler, layer)


        super().__init__(              
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )


class Linker(CLFlayer):
    """
    Layer which works on segment level
    """

    def __init__(self, 
                layer:Union[nn.Module, str], 
                hyperparams:dict, 
                input_size:int,
                output_size:int=None,
                ):
        self.task = "link"
        self.level = "seg"


        if isinstance(layer, str):
            layer = getattr(linkers, layer)


        super().__init__(              
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )


class LinkLabeler(CLFlayer):
    """
    Layer which works on segment level
    """

    def __init__(self, 
                layer:Union[nn.Module, str], 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                ):

        self._return_link_preds = False
        if layer in ["DirLinkLabeler"]:
            self._return_link_preds = True

        self.task = "link_label"
        self.level = "seg" if layer != "DirLinkLabeler" else "p_seg"

        if isinstance(layer, str):

            try:
                layer = getattr(general, layer)
            except:
                layer = getattr(link_labelers, layer)


        super().__init__(              
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )

    def forward(self, *args, **kwargs):

        if self._return_link_preds:
            logits, link_label_preds, link_preds = self.layer(*args, **kwargs)

            return [
                    {
                    "task": self.task,
                    "level": self.level,
                    "logits": logits,
                    "preds": link_label_preds,
                    },
                    {
                    "task": "link",
                    "level": self.level,
                    "preds": link_preds,
                    },       
                    ]

        else:
            return super().forward(*args, **kwargs)