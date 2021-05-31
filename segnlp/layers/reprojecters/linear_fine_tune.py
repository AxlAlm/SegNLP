

#pytroch
import torch.nn as nn
from torch import Tensor


class LinearRP(nn.Module):


    def __init__(self, input_size:int, hidden_size:int=None, dropout:float = 0.0):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)

        if hidden_size is None:
            hidden_size = input_size

        self.fine_tune = nn.Linear(input_size, hidden_size)
        self.output_size = hidden_size


    def forward(self, input:Tensor):
        input = self.dropout(input)
        return self.fine_tune(input)



