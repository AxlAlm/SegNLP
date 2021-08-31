
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
        self.vocab = vocab
        self.output_size = len(self.vocab)

        self.reduce_dim = dim
        if self.reduce_dim is not None:
            self.dim_reduction =  nn.Linear(self.output_size, dim)
            self.output_size = dim                  


    def forward(self, input:Tensor, span_idxs:Tensor):

        batch_size = input.shape[0]
        _, seg_size, *_ = span_idxs.shape

        # one hots
        batch_bow = torch.zeros(batch_size, seg_size, self.output_size, device=input.device)

        # k = sample idx, s = segment index, i = segment start index, j = segment end index,
        for k in range(batch_size):
            for s, (i,j) in enumerate(span_idxs[k]):
                
                #encode
                token_ids = self.vocab[input[k][i:j]] 

                # fill one hots
                batch_bow[k][s][token_ids] = 1

        if self.reduce_dim:
            return self.dim_reduction(batch_bow)
        else:
            return batch_bow