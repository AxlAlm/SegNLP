import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class CBAttentionLayer(nn.Module):

    """
    based on:
    https://arxiv.org/pdf/1506.03134v1.pdf


    What we are doing:

    We are getting the probabilites across all units in a sample given one of the units. This tells us which 
    unit is important given a unit. We will use this score to treat it as a pointer. .e.g. this units points to 
    the index of the unit where which given the highest score.

    so,

    query is a representations of a unit at position i
    key is representation for all units in the a sample

    1)
    each of these are passed to linear layers so they are trained.

    2)
    then we add the quary to the key. I.e. we add the representation of the unit at i to all units in our sample.

    3)
    Now we have a representation which at each index represent the query unit and the unit at the position n

    4)
    then we pass this into tanh() then linear layer so we can learn the attention,

    5)
    Then we apply softmax to get the final attention scores.

   

    """

    def __init__(   
                    self,
                    input_dim:int,
                    ):

        super().__init__()
        self.W1 = nn.Linear(input_dim, input_dim)
        self.W2 = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1)
        

    def forward(self, query, key, mask=None, return_softmax=False):

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
        
        #for all samples we set the probs for non existing units to inf and the prob for all
        # units pointing to an non existing unit to -inf.
        mask = mask.type(torch.bool)
        ui[~mask]  =  float("-inf") #set full units vectors to -inf
    
        if return_softmax:
            return  F.softmax(ui, dim=-1)
        else:
            return ui
