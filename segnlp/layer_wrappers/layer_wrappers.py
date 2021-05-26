#basics
import inspect

#pytorch
import torch.nn as nn
from torch import Tensor
import torch

#segnlp
from segnlp.utils import BIODecoder
from segnlp.layers import encoders 
from segnlp.layers import embedders
from segnlp.layers import reducers
from segnlp.layers import linkers
from segnlp.layers import segmenters
from segnlp.layers import general


class Layer(nn.Module):

    def __init__(self, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int, 
                output_size:int=None,
                ):
        super().__init__()
        input_size = input_size
        #output_size = output_size

        params = hyperparams
        params["input_size"] = input_size
        params["output_size"] = output_size

        print(params)
        params = self.__filter_paramaters(layer, params)
        print(params)

        self.layer = layer(**params)

        if hasattr(self.layer, "output_size"):
            self.output_size = self.layer.output_size
    

    def __filter_paramaters(self, layer, params):
        sig = inspect.signature(layer)

        filtered_params = {}
        for para in sig.parameters:

            if para in params:
                filtered_params[para] = params[para]
        
        return filtered_params


    def _call(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


    def forward(self, **kwargs):
        #kwargs = self.__filter_paramaters(**kwargs)
        out = self._call(**kwargs)

        # assert isinstance(output, dict)
        # assert torch.is_tensor(loss)
        return  out  


class Embedder(Layer):

    def __init__(self, layer:nn.Module, hyperparams:dict, input_size:int):
        
        if isinstance(layer, str):
            layer = getattr(embedders, layer)

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


    def loss(self, target:Tensor, logits:Tensor):

        if hasattr(self.layer, "loss"):
            loss = self.layer.loss(logits=logits, **kwargs)
        else:

            if self.prediction_level == "unit":
                return F.cross_entropy(
                                        torch.flatten(logits, end_dim=-2), 
                                        target.view(-1), 
                                        reduction="mean",
                                        ignore_index=-1
                                        )
            else:
                raise NotADirectoryError

        return loss, output



    def _call(self, *args, **kwargs):
        logits, output =  self.layer(*args, **kwargs)

        if not self.inference:
            loss = self.loss(target="", logits=logits)
            return loss, output

        return output


class Segmenter(CLFlayer):
    """
    Layer which works on token level
    """

    def __init__(self, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                labels:dict,
                encoding_scheme:str="bio",
                ):

        if isinstance(layer, str):
            layer = getattr(segmenters, layer)

        super().__init__(
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )

        if encoding_scheme == "bio":
            self.seg_decoder = BIODecoder(
                                        B = [i for i,l in enumerate(labels) if "B-" in l],
                                        I = [i for i,l in enumerate(labels) if "I-" in l],
                                        O = [i for i,l in enumerate(labels) if "O-" in l],
                                        )
        else:
            raise NotImplementedError(f'"{encoding_scheme}" is not a supported encoding scheme')


    def _call(self, *args, **kwargs):
        logits, output =  self.layer(*args, **kwargs)

        output.update(self.seg_decoder(
                                        batch_encoded_bios = output["preds"], 
                                        lengths = kwarg["lengths"],                  
                                        ))

        if not self.inference:
            loss = self.loss(target="", logits=logits)
            return loss, output

        return output


class Labeler(CLFlayer):
    """
    Layer which works on segment level
    """

    def __init__(self, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                ):

        if isinstance(layer, str):
            #layer = getattr(linkers, layer)
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
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                ):

        if isinstance(layer, str):
            layer = getattr(linkers, layer)

        super().__init__(              
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )




# class LinkLayer(Layer):
#     """
#     Layer which works on segment level
#     """

#     def __init__(self, 
#                 layer:nn.Module, 
#                 hyperparams:dict, 
#                 input_size:int,
#                 output_size:int,
#                 ):

#         if isinstance(layer, str):
#             layer = getattr(linkers, layer)


#         super().__init__(              
#                         layer=layer, 
#                         hyperparams=hyperparams,
#                         input_size=input_size,
#                         output_size=output_size
#                         )


#     def loss(self, target:Tensor, logits:Tensor):

#         if self.prediction_level == "unit":
#             self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)
#             return self.loss(torch.flatten(logits, end_dim=-2), target.view(-1))
#         else:
#             raise NotADirectoryError
            

#     def _call(self, *args, **kwargs):
#         logits, output =  self.layer(*args, **kwargs)

#         if not self.inference:
#             loss = self.layer.loss(logits=logits, **kwargs)
#             return loss, output

#         return output