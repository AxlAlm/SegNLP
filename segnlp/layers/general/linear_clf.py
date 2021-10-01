
#basics
from typing import Union


#pytorch
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

#segnlp
from segnlp import utils

class LinearCLF(nn.Module):

    def __init__(self, 
                input_size, 
                output_size, 
                loss_reduction = "mean",
                ignore_index = -1,
                weight_init : Union[str, dict] = None,
                ):
        super().__init__()
        self.clf = nn.Linear(input_size, output_size)
        self.loss_reduction = loss_reduction
        self.ignore_index = ignore_index
        utils.init_weights(self, weight_init)

          
    def forward(self, input:Tensor):

        logits = self.clf(input)
        preds = torch.argmax(logits, dim=-1)

        return logits, preds


    def loss(self, logits:Tensor, targets:Tensor):
        return F.cross_entropy(
                                torch.flatten(logits, end_dim=-2), 
                                targets.view(-1), 
                                reduction = self.loss_reduction,
                                ignore_index = self.ignore_index
                                )
