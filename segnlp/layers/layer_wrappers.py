#basics
import inspect

#pytorch
import torch.nn as nn
from torch import Tensor
import torch

#segnlp
from segnlp.nn.utils import BIODecoder
from segnlp.nn.layers.token_loss import TokenCrossEntropyLoss


class Layer(nn.Module):

    def __init__(self, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int, 
                output_size:int=None,
                ):
        super().__init__()
        input_size = input_size
        output_size = output_size

        params = hyperparams
        params["input_size"] = input_size
       
        if output_size is not None:
            params["output_size"] = output_size

        if isinstance(layer, str):
            pass
        else:
            self.layer = layer(**params)

        self.output_size = self.layer.output_size
    

    def forward(self, *args, **kwargs):
        loss, output = self._call(*args, **kwargs)

        assert isinstance(output, dict)
        assert torch.is_tensor(loss)

        return  loss, output     


class RepLayer(Layer):

    def __init__(self, layer:nn.Module, hyperparams:dict, input_size:int):
        super().__init__(layer=layer, hyperparams=hyperparams, input_size=input_size)
        

    def _call(self, input:Tensor, batch:dict):
        return self.layer(
                        input=input,
                        batch=batch
                        )


class TokenLayer(Layer):
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
        logits, preds =  self.layer(*args, **kwargs)
        output.update(self.seg_decoder(
                                        batch_encoded_bios = preds, 
                                        lengths = kwarg["lengths"],                  
                                        ))

        if not self.inference:
            self.layer.loss(logits=logits, **kwargs)

        return loss, output



class SegLayer(Layer):
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


    def _call(self, *args, **kwargs):
        logits, preds =  self.layer(*args, **kwargs)

        if not self.inference:
            self.layer.loss(logits=logits, **kwargs)

        return loss, output
