

#basics
import numpy as np
from typing import Union


#pytorch
import torch.nn as nn


def init_weights(module: nn.Module, weight_init: Union[str, dict]):
    """
    Will init all linear layers in a nn.Module
    """

    # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
    

    if weight_init is None:
        return

    if isinstance(weight_init, dict):
        init_method = weight_init.pop("name")
        kwargs = weight_init

    if isinstance(weight_init, str):
        init_method = weight_init
        kwargs = {}


    for name, param in module.named_parameters():

        if "weight" in name:

            # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
            if init_method == "normal":
                kwargs["std"] = 1/np.sqrt(m.in_features)
                getattr(nn.init, "normal_")(param.data, **kwargs)
                
            else:
                getattr(nn.init, init_method)(param.data, **kwargs)
        
        if "bias" in name:
            param.data.fill_(0)