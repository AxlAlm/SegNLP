



# basics
from typing import List, Tuple, DefaultDict, Dict


# pytorch
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class PairDirLabeler(nn.Module):


    def __init__(self, input_size:int, hidden_size:int, output_size:int, dropout:float=0.0):
        super().__init_()

        output_size = ((output_size -1) * 2 ) + 1
        self.link_label_clf_layer = nn.Sequential(
                                nn.Linear(input_size, hidden_size),
                                nn.Tanh(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_size, output_size),
                            )


    def forward(self, input:Tensor):

        logits = self.link_label_clf_layer(input)
