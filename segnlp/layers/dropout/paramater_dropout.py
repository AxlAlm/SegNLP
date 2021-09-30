








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
        Removes, for each sample, unique paramaters/features and amplifies others. I.e. if our features are word embeddings we 
        remove unique rows in all the words embeddings in the sample and amplifies others.

        e.g. if the following tensor is a sample of 2 words:

            tensor([[0.3205, 0.1266, 0.3539, 0.5185],
                    [0.9885, 0.3204, 0.9636, 0.4875]])

        and example output could be:

            tensor([[0.6410, 0.2533, 0.0000, 0.0000],
                    [1.9770, 0.6409, 0.0000, 0.0000]])


        if p == 0.5, non masked values will be multipled by 2

        from:
        https://github.com/flairNLP/flair/blob/master/flair/nn/dropout.py

        """

        if not self.training or not self._p:
            return input

        # create a zero tensor with the shape ( batch_size, 1 , feature_dim)
        mask = torch.zeros((input.size(0), 1, input.size(1)), device = input.device, dtype = torch.long)

        # fill the mask with 1s based on bernoulli, i.e. for each value in the tensor we set
        # set value to either 1 based on prob p and to 0 based on q.
        # NOTE! a higher dropout value means that more words should be masked out, which means
        # we need to use q and not p, which is calcualted q = 1 - p        
        q = 1 - self._p
        mask = mask.bernoulli_(q)

        # amplification
        mask /= q

        return input * mask