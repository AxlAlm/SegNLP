


# basics
from typing import Union

# pytorch
from torch import Tensor
import torch.nn as nn
import torch


class AdditiveAttention(nn.Module):

    """
    Originally from:
        https://arxiv.org/pdf/1409.0473v5.pdf

    Also referenced to as Content Based Attention:
        https://arxiv.org/pdf/1506.03134v1.pdf

    Attention is learned for a query vector over a set of vectors. 
    If we have a query vector and then a key matrix with the size (n,k), 
    we will create attention vector over n

    
    NOTE!
    We mask out the attention scores for the positions which its impossible for 
    the segments to attend to. I.e. the padding.


    """

    def __init__(   
                    self,
                    input_dim:int,
                    ):

        super().__init__()
        self.W1 = nn.Linear(input_dim, input_dim)
        self.W2 = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1)
        

    def forward(self, query:Tensor, key:Tensor, mask:Tensor, softmax = False):

        # key (also seens as context) (BATCH_SIZE, SEQ_LEN, INPUT_DIM )
        key_out = self.W1(key)
    
        # query = (BATCH_SIZE, 1, INPUT_DIM )
        query_out = self.W2(query.unsqueeze(1))
        
        #(BATCH_SIZE, SEQ_LEN, INPUT_DIM*2)
        ui = self.v(torch.tanh(key_out + query_out)).squeeze(-1)
        
        # masking needs to be done so that the sequence length
        # of each sequence is adhered to. I.e. the non-padded
        # part needs to amount to 1, not the full sequence with padding.
        # apply it like this :
        # http://juditacs.github.io/2018/12/27/masked-attention.html
        
        #for all samples we set the probs for non existing segments to inf and the prob for all
        # segments pointing to an non existing segment to -inf.
        mask = mask.type(torch.bool)
        ui[~mask]  =  float("-inf") #set full segments vectors to -inf


        if softmax:
            ui = torch.softmax(ui)

        return ui


