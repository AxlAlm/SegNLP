
#pytorch
import torch
import torch.nn as nn
from torch import Tensor


class LinearCLF(nn.Module):

    def __init__(self, input_size, output_size, dropout:float=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(input_size, output_size)

    def forward(self, input:Tensor):
        return self.clf(self.dropout(input))
