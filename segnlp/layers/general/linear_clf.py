
#pytorch
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F



class LinearCLF(nn.Module):

    def __init__(self, 
                input_size, 
                output_size, 
                dropout:float=0.0,
                loss_reduction = "mean",
                ignore_index = -1,
                weight_init = None,
                ):
        super().__init__()
        self.loss_reduction = loss_reduction
        self.ignore_index = ignore_index
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(input_size, output_size)

        self.__weight_init(weight_init = weight_init)

    
    def __weight_init(self, weight_init:str):
        
        if weight_init is None:
            torch.nn.init.uniform_(self.clf.weight.data,  a=-0.05, b=0.05)
            torch.nn.init.uniform_(self.clf.bias.data,  a=-0.05, b=0.05)
        else:
            raise NotImplementedError()


    def forward(self, input:Tensor):

        logits = self.clf(self.dropout(input))
        preds = torch.argmax(logits, dim=-1)

        return logits, preds


    def loss(self, logits:Tensor, targets:Tensor):
        return F.cross_entropy(
                                torch.flatten(logits, end_dim=-2), 
                                targets.view(-1), 
                                reduction = self.loss_reduction,
                                ignore_index = self.ignore_index
                                )