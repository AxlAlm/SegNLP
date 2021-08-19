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
from segnlp.layers import embedders
from segnlp.layers import reducers
from segnlp.layers import linkers
from segnlp.layers import segmenters
from segnlp.layers import general
from segnlp.layers import reprojecters
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

        if hasattr(self.layer, "output_size"):
            self.output_size = self.layer.output_size
        

        self.frozen = False
        if params.get("freeze", False):
            self.__freeze()
            self.frozen = True


    def __freeze(self):
        for param in self.layer.parameters():
            param.requires_grad = False


    # def __filter_paramaters(self, layer, params):
    #     sig = inspect.signature(layer)

    #     filtered_params = {}
    #     for para in sig.parameters:

    #         if para in params:
    #             filtered_params[para] = params[para]
        
    #     return filtered_params


    def forward(self, *args, **kwargs):
        #kwargs = self.__filter_paramaters(**kwargs)
        return self.layer(*args, **kwargs)



class Embedder(Layer):

    def __init__(self, layer:nn.Module, hyperparams:dict):
        
        if isinstance(layer, str):
            layer = getattr(embedders, layer)

        super().__init__(layer=layer, hyperparams=hyperparams)
        

class Reprojecter(Layer):


    def __init__(self, layer:nn.Module, hyperparams:dict, input_size:int):
        
        if isinstance(layer, str):
            layer = getattr(reprojecters, layer)

        super().__init__(layer=layer, hyperparams=hyperparams, input_size=input_size)


class Encoder(Layer):

    def __init__(self, layer:nn.Module, hyperparams:dict, input_size:int):
        
        if isinstance(layer, str):
            layer = getattr(encoders, layer)

        super().__init__(layer=layer, hyperparams=hyperparams, input_size=input_size)
        

class Reducer(Layer):
    
    def __init__(self, layer:nn.Module, hyperparams:dict, input_size:int):
        
        if isinstance(layer, str):
            layer = getattr(reducers, layer)

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


    # def loss(self, *args, **kwargs):

    #     if hasattr(self.layer, "loss"):
    #         loss = self.layer.loss(*args, **kwargs)
    #     else:

    #         if isinstance(self, Segmenter):
    #             raise NotImplementedError

    #         else:
    #             logits = kwargs["logits"]
    #             targets = kwargs["targets"]
    #             loss = F.cross_entropy(
    #                                     torch.flatten(logits,end_dim=-2), 
    #                                     targets.view(-1), 
    #                                     reduction="mean",
    #                                     ignore_index=-1
    #                                     )
            

    #     return loss


    def _call(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class Segmenter(CLFlayer):
    """
    Layer which works on token level
    """

    def __init__(self, 
                layer:Union[nn.Module, str], 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                #task:str,
                ):
        # self.task = task
        # self.level = "token"

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
        # self.task = "label"
        # self.level = "seg"

        if isinstance(layer, str):
            #layer = getattr(Labeler, layer)
            layer = getattr(general, layer)


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
                output_size:int,
                ):
        # self.task = "link"
        # self.level = "seg"

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
        # self.task = "link_label"
        # self.level = "seg"

        if isinstance(layer, str):
            layer = getattr(link_labelers, layer)
            #layer = getattr(general, layer)

        super().__init__(              
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )