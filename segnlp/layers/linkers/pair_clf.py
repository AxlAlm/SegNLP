
import numpy as np

#pytorch
import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn

#segnlp
from segnlp.layers.general import LinearCLF


class PairCLF(LinearCLF):

    """
    Input is assumed to be pair embedding.
  
    1 ) pass input to linear layer

    2) set the values for all pairs which are not possible and which we dont want to be counted for in our
        loss function  to -inf. All pairs across padded segments are set to -inf

    """

    def __init__(self,
                input_size:int, 
                dropout:float=0.0,
                loss_reduction = "mean",
                ignore_index = -1,
                weight_init = "normal",
                weight_init_kwargs : dict = {}):
        super().__init__(
                        input_size = input_size, 
                        output_size = 1, 
                        dropout = dropout,
                        loss_reduction = loss_reduction,
                        ignore_index = ignore_index,
                        weight_init = weight_init,
                        weight_init_kwargs = weight_init_kwargs
                        )

    def forward(self, 
                input:torch.tensor, 
                segment_mask:torch.tensor, 
                ):

        #predict links
        pair_logits = self.clf(self.dropout(input)).squeeze(-1)
    
    
        # for all samples we set the probs for non existing segments to inf and the prob for all
        # segments pointing to an non existing segment to -inf.
        pair_logits[~segment_mask]  =  float("-inf")
        pf = torch.flatten(pair_logits, end_dim=-2)
        mf = torch.repeat_interleave(segment_mask, segment_mask.shape[1], dim=0)
        pf[~mf] = float("-inf")
        logits = pf.view(pair_logits.shape)
        preds = torch.argmax(logits, dim=-1)

        return logits, preds 



    def loss(self, logits:Tensor, targets:Tensor):
        return F.cross_entropy(
                                torch.flatten(logits, end_dim=-2), 
                                targets.view(-1), 
                                reduction = self.loss_reduction,
                                ignore_index = self.ignore_index
                                )

