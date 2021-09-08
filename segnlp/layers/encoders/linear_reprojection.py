

#basics
import numpy as np

#pytroch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from segnlp import utils


class LinearRP(nn.Module):

    """
    Reprojects input via linear layer. Its simply a wrapper around a linear layer
    with options to use dropout and activation function. 
    
    """

    def __init__(self, 
                    input_size:int, 
                    hidden_size:int=None, 
                    activation:str=None,
                    dropout:float = 0.0,
                    weight_init: str = "normal",
                    weight_init_kwargs: dict = {}
                    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)

        if hidden_size is None:
            hidden_size = input_size


        if activation is not None:
            self.reproject = nn.Sequential(
                                        nn.Linear(input_size, hidden_size),
                                        getattr(nn, activation)()
                                        )
        else:
            self.reproject = nn.Linear(input_size, hidden_size)

        self.apply(utils.get_weight_init_fn(weight_init, weight_init_kwargs))
        self.output_size = hidden_size


    def forward(self, input:Tensor):
        input = self.dropout(input)
        return self.reproject(input)
        

