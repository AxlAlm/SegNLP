

#basics
import numpy as np
from functools import partial


#pytorch
import torch
import torch.nn as nn


def get_weight_init_fn(init_method:str , kwargs:dict = {}):
    """
    Wrapped function for init pytorch weights
    """

    # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
    
    
    def fn(m):
        
        if type(m) != nn.Linear:
            return
        
        # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        if init_method == "normal":
            kwargs["std"] = 1/np.sqrt(m.in_features)
            getattr(nn.init, "normal_")(m.weight.data, **kwargs)
        else:
            getattr(nn.init, init_method)(m.weight.data, **kwargs)
            
        m.bias.data.fill_(0)


        
    return fn