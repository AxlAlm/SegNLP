
# basics
from typing import Sequence
import numpy as np

#pytroch
import torch.nn as nn
from torch import Tensor
import torch

#segnlp
from segnlp.resources.vocab import Vocab
from segnlp import utils


class SegBOW(nn.Module):

    def __init__(   
                self,
                vocab: Vocab,
                #out_dim : int = None,
                dropout: float = 0.0,
                mode: str = "one_hot",
                ):
        super().__init__()

        assert mode in ["one_hot", "count"]

        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.output_size = self.vocab_size #out_dim if out_dim is not None else self.vocab_size
        self.mode = mode

        # if out_dim is not None:
        #     self.reduce_dim = nn.Linear(len(self.vocab), self.output_size)

        self.dropout = nn.Dropout(dropout)


    def forward(self, input:Sequence[str], lengths:Tensor, span_idxs:Tensor):
        
        batch_size, seg_size, *_ = span_idxs.shape

        sample_tokens = np.split(input, utils.ensure_numpy(torch.cumsum(lengths, dim = 0)))[:-1] # np.split creates a empty tail
        
        assert len(sample_tokens) == len(lengths)
   
        # one hots
        bow = torch.zeros(
                            (batch_size, seg_size, self.vocab_size),
                            device = span_idxs.device,
                            dtype = torch.int64,
                            )
    
        # k = sample idx, s = segment index, i = segment start index, j = segment end index,
        for k in range(batch_size):

            for s, (i,j) in enumerate(span_idxs[k]):

                if i == 0 and j == 0:
                    continue

                # encode words
                token_ids = self.vocab[sample_tokens[k][i:j]] 

                if self.mode == "one_hot":
                    # fill one hots
                    bow[k][s][token_ids] = 1
                else:
                    bow[k][s][token_ids] += 1


        # if hasattr(self, "reduce_dim"):
        #     bow = self.reduce_dim(bow.type(torch.float))
        
        bow = self.dropout(bow)

        return bow
        
        