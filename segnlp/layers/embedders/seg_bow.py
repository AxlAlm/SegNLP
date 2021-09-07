
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
                mode: str = "one_hot",
                ):
        super().__init__()
        self.vocab = vocab
        self.output_size = len(self.vocab)
        self.mode = mode # will support Counts later


    def forward(self, input:Sequence, lengths:Tensor, span_idxs:Tensor):
        
        batch_size, seg_size, *_ = span_idxs.shape
        sample_tokens = np.split(input, utils.ensure_numpy(lengths))
        
        # one hots
        bow = torch.LongTensor(batch_size, seg_size, self.output_size, device=span_idxs.device)

        # k = sample idx, s = segment index, i = segment start index, j = segment end index,
        for k in range(batch_size):
            for s, (i,j) in enumerate(span_idxs[k]):
                
                #encode
                token_ids = self.vocab[sample_tokens[k][i:j]] 

                if self.mode == "one_hot":
                    # fill one hots
                    bow[k][s][token_ids] = 1
                else:
                    raise NotImplementedError

        return bow