
#pytorch
import torch
import torch.nn as nn
from torch import Tensor


class LinearCLF(nn.Module):

    def __init__(self, input_size, output_size, dropout:float=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(input_size, output_size)

    
    def __weight_init(self, weight_init:str):
        
        if weight_init is None:
            torch.nn.init.uniform_(self.link_clf.weight.data,  a=-0.05, b=0.05)
            torch.nn.init.uniform_(self.link_clf.bias.data,  a=-0.05, b=0.05)
        else:
            raise NotImplementedError()


    def forward(self, input:Tensor):

        logits = self.clf(self.dropout(input))
        preds = torch.argmax(logits, dim=-1)

        return logits, preds