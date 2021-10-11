
# basics
from typing import Sequence, Union
import numpy as np

#pytroch
import torch.nn as nn
from torch import Tensor
import torch

#segnlp
from segnlp.resources import vocabs
from segnlp import utils


class SegBOW(nn.Module):

    """
    Creates BOW features vectors for segments.

    Features can either be one-hot encodings or count vectors
    """

    def __init__(   
                self,
                vocab: Union[str, list],
                mode: str = "one_hots",
                ):
        super().__init__()

        assert mode in ["one_hots", "counts"]

        self.vocab = getattr(vocabs, vocab)() if isinstance(vocab, str) else vocabs.Vocab(list)
        self.vocab_size = len(self.vocab)
        self.output_size = self.vocab_size #out_dim if out_dim is not None else self.vocab_size
        self.mode = mode


    def forward(self, 
                input:Sequence[str], 
                lengths:Tensor, 
                span_idxs:Tensor,
                device : Union[str, torch.device] = "cpu"
                ):
        
        batch_size, seg_size, *_ = span_idxs.shape

        sample_tokens = np.split(input, utils.ensure_numpy(torch.cumsum(lengths, dim = 0)))[:-1] # np.split creates a empty tail
           
        # one hots
        bow = torch.zeros(
                            (batch_size, seg_size, self.vocab_size),
                            dtype = torch.int64,
                            )
    
        # k = sample idx, s = segment index, i = segment start index, j = segment end index,
        for k in range(batch_size):

            for s, (i,j) in enumerate(span_idxs[k]):

                if i == 0 and j == 0:
                    continue

                # encode words
                token_ids = self.vocab[sample_tokens[k][i:j]] 

                if self.mode == "one_hots":
                    # fill one hots
                    bow[k][s][token_ids] = 1
                else:
                    bow[k][s][token_ids] += 1


        return bow.type(torch.float).to(device)
        
        