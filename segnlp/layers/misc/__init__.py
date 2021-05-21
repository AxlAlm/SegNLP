

# pytorch
import torch
import torch.nn as nn
from torch import Tensor


class LinearFT(nn.Module):

    def __init__(self, 
                input_size:int, 
                output_size:int,
                #activation:str=None,
                dropout:float=0.0,
                )

        self.dropout = nn.Dropout(dropout)
        self.ft = nn.Linear(input_size,output_size)

        # self.activation = None
        # if self.activation is not None:
        #     self.activation = getattr(nn, activation)

    def forward(self, input:Tensor):
        return self.ft(self.dropout(input))