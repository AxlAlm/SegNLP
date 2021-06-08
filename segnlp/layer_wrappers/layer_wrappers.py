#basics
import inspect

#pytorch
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from segnlp.utils import BIODecoder
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


    def __filter_paramaters(self, layer, params):
        sig = inspect.signature(layer)

        filtered_params = {}
        for para in sig.parameters:

            if para in params:
                filtered_params[para] = params[para]
        
        return filtered_params


    def _call(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


    def forward(self, *args, **kwargs):
        #kwargs = self.__filter_paramaters(**kwargs)
        return self._call(*args, **kwargs)
  
        # assert isinstance(output, dict)
        # assert torch.is_tensor(loss)
        #return  out  



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


    def loss(self, *args, **kwargs):

        if hasattr(self.layer, "loss"):
            loss = self.layer.loss(*args, **kwargs)
        else:

            if isinstance(self, Segmenter):
                raise NotImplementedError

            else:
                logits = kwargs["logits"]
                targets = kwargs["targets"]
                loss = F.cross_entropy(
                                        torch.flatten(logits,end_dim=-2), 
                                        targets.view(-1), 
                                        reduction="mean",
                                        ignore_index=-1
                                        )
            

        return loss


    def _call(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class Segmenter(CLFlayer):
    """
    Layer which works on token level
    """

    def __init__(self, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                task:str = None, 
                decode:bool = False,
                encoding_scheme:str = "bio",
                labels:dict=None,
                ):
        self.task = task

        if isinstance(layer, str):
            layer = getattr(segmenters, layer)

        self.schedule = None
        if "scheduler" in hyperparams:
            self.schedule = ScheduleSampling(
                                            schedule="inverse_sig",
                                            k=hyperparams["k"])

        super().__init__(
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )

        self.decode = decode
        if decode:
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

        if self.schedule is not None:

            batch = kwargs.pop("batch")
            if self.schedule.next(batch.current_epoch):
                preds = batch["token"][self.task]


        if self.decode:
            seg_data = self.seg_decoder(
                                            batch_encoded_bios = preds, 
                                            lengths = kwargs["lengths"],                  
                                            )
            return logits, preds, seg_data

        else:
            return logits, preds, {}


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
            layer = getattr(linkers, layer)
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



class LinkLabeler(CLFlayer):
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
            layer = getattr(link_labelers, layer)
            layer = getattr(general, layer)

        super().__init__(              
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )