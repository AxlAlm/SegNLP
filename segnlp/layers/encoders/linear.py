

#basics
import numpy as np

#pytroch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from segnlp import utils


class Linear(nn.Module):

    """
    Reprojects input via linear layer. Its simply a wrapper around a linear layer
    with options to use dropout and activation function. 
    
    """

    def __init__(self, 
                    input_size:int, 
                    hidden_size:int=None, 
                    activation:str=None,
                    weight_init: str = "normal",
                    weight_init_kwargs: dict = {}
                    ):
        super().__init__()
        
        if hidden_size is None:
            hidden_size = input_size

        if activation is not None:
            self.linear = nn.Sequential(
                                        nn.Linear(input_size, hidden_size),
                                        getattr(nn, activation)()
                                        )
        else:
            self.linear = nn.Linear(input_size, hidden_size)

        self.apply(utils.get_weight_init_fn(weight_init, weight_init_kwargs))
        self.output_size = hidden_size


    def forward(self, input:Tensor):
        return self.linear(input)
        


