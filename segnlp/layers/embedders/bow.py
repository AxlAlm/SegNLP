
#pytroch
import torch.nn as nn
from torch import Tensor
import torch

#segnlp
from segnlp.resources.vocab import Vocab


class BOW(nn.Module):

    def __init__(   
                self,
                vocab: Vocab,
                dim:int=None,
                ):
        super().__init__()
        self.output_size = vocab.size

        self.reduce_dim = dim
        if self.reduce_dim is not None:
            self.dim_reduction =  nn.Linear(self.output_size, dim)
            self.output_size = dim                  


    def forward(self, word_encs:Tensor, span_idxs:Tensor):

        batch_size, nr_tokens = word_encs.shape
        _, seg_size, *_ = span_idxs.shape

        batch_bow = torch.zeros(batch_size, seg_size, self.output_size, device=word_encs.device)

        for k in range(batch_size):
            for s, (i,j) in enumerate(span_idxs[k]):
                word_ids = word_encs[k][i:j]
                batch_bow[k][s][word_ids] = 1

        if self.reduce_dim:
            return self.dim_reduction(batch_bow)
        else:
            return batch_bow