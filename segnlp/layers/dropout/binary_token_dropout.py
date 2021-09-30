



#basics
from typing import Union, List


# pytroch
import torch
from torch import nn
from torch.functional import Tensor


class BinaryTokenDropout(nn.Module):

    def __init__(self, p: float):
        super().__init__()
        self._p = p

    def forward(self, input : Tensor) -> Tensor:

        """
        Removes token embeddings from input. I.e. turns all values in random vectors on dim = 2 to 0

        from:
        https://github.com/flairNLP/flair/blob/master/flair/nn/dropout.py

        """
        
        if not self.training or not self._p:
            return input

        # create a zero tensor with the shape ( batch_size, nr_words)
        mask = torch.zeros((input.size(0), input.size(1)), device = input.device, dtype = torch.long)

        # fill the mask with 1s based on bernoulli, i.e. for each value in the tensor we set
        # set value to either 1 based on prob p and to 0 based on q.
        # NOTE! a higher dropout value means that more words should be masked out, which means
        # we need to use q and not p, which is calcualted q = 1 - p 
        q = 1 - self.p  
        mask = mask.bernoulli_(q)

        # then we multiply the mask with the input tensor. We unsqueeze be able to multiply each 
        # word vector (emb) with a vector of its own, e.g. we cannot do instead of emb * 1 but we can do emb * [1]
        return input * mask.unsqueeze(-1)
