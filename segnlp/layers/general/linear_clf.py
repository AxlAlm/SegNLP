
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
                dropout:float=0.0,
                loss_reduction = "mean",
                ignore_index = -1,
                weight_init = "normal",
                weight_init_kwargs : dict = {}
                ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(input_size, output_size)

        self.loss_reduction = loss_reduction
        self.ignore_index = ignore_index
        self.apply(utils.get_weight_init_fn(weight_init, weight_init_kwargs))


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