
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, 
                input:torch.tensor, 
                segment_mask:torch.tensor, 
                ):

        #predict links
        pair_logits, _ = self.clf(input)

        pair_logits = pair_logits.squeeze(-1)

        # for all samples we set the probs for non existing segments to inf and the prob for all
        # segments pointing to an non existing segment to -inf.
        segment_mask = segment_mask.type(torch.bool)
        pair_logits[~segment_mask]  =  float("-inf")
        pf = torch.flatten(pair_logits, end_dim=-2)
        mf = torch.repeat_interleave(segment_mask, segment_mask.shape[1], dim=0)
        pf[~mf] = float("-inf")
        logits = pf.view(pair_logits.shape)
        preds = torch.argmax(logits, dim=-1)

        return logits, preds 
