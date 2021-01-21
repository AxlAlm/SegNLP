
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class CONTENT_BASED_ATTENTION(nn.Module):

    """
    based on:
    https://arxiv.org/pdf/1612.08994.pdf

    """

    def __init__(   
                    self,
                    input_dim:int,
                    ):

        super().__init__()
        self.Q = nn.Linear(input_dim, input_dim)
        self.C = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1)
        

    def forward(self, query, context, mask=None):

        # query = (BATCH_SIZE, 1, INPUT_DIM )
        query_out = self.Q(query.unsqueeze(1))

        # contex_out = (BATCH_SIZE, SEQ_LEN, INPUT_DIM )
        contex_out = self.C(context)

        #(BATCH_SIZE, SEQ_LEN)
        u = self.v(F.tanh(query_out + contex_out)).squeeze(-1)
        
        # masking needs to be done so that the sequence length
        # of each sequence is adhered to. I.e. the non-padded
        # part needs to amount to 1, not the full sequence with padding.
        # apply it like this :
        # http://juditacs.github.io/2018/12/27/masked-attention.html
        
        if mask.dtype != torch.bool:
            mask = mask.type(torch.bool)
        
        u[~mask] = float('-inf')
    
        # (BATCH_SIZE, SEQ_LEN)
        scores = F.softmax(u)

        return scores




