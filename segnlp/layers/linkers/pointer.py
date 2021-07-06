#basics
import numpy as np

#pytorch
from torch import Tensor
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
    Then we mask out the attention scores for the positions which its impossible for the units to attend to. I.e. the padding.


    """

    def __init__(   
                    self,
                    input_dim:int,
                    ):

        super().__init__()
        self.W1 = nn.Linear(input_dim, input_dim)
        self.W2 = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1)
        

    def forward(self, query, key, mask=None):

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
    
        return ui



class Pointer(nn.Module):

    """

    Implementation of 

    1) https://arxiv.org/pdf/1612.08994.pdf
    
    
    more about pointers
    2) https://arxiv.org/pdf/1506.03134v1.pdf


    NOTE!

    In the original paper (1), there is little information about what the decoder does more than applying Content Based Attention.
    In Figure 3 its implied that FC3 is applied at each timestep of the LSTM, except the first. 
    However, some reasons suggests this is incorrect;

    1) they state that the reason for FC is to reduce the dimension of a sparse input, the output of the LSTM is not sparse. So,
        when they talk about input to LSTM they must talk about the input to ANY of the LSTM (encoder or decoder)
     
    2) a Seq2seq network takes input at both encoding and decoding, meaning that the input to the decoding LSTM needs to be 
        something and what makes most sense is that its the sparse features passed to FC3 then to the LSTM.

    3) they state in the variations of models they test that they test:
    
        
            "; 4) A non-sequence-tosequence model that uses the hidden layers produced by the BLSTM encoder with the same type
            of attention as the joint model (called Joint Model No Seq2Seq in the table). That is, di n Equation 3 is replaced by ei.

        meaning that they do not create a decoding representation but use the encoding representations for the decoding. This
        suggests the input to the decoder is the feature representation to FC3 then to LSTM

    
    4) in the Pointer paper (2), they pass inputs to the decoder and encoder. Following this both FC1 and FC3 would need to 
        first reduce the dim of the features if an input to be able to be passed to the encoder and decoder.

    """


    def __init__(self, input_size:int):
        super().__init__()
        self.attention = CBAttentionLayer(
                                        input_dim=input_size,
                                        )

    def forward(self, inputs:Tensor, encoder_outputs:Tensor, mask:Tensor):

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        device = inputs.device

        logits = torch.zeros(batch_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            logits[:, i] = self.attention(input[:,i], encoder_outputs, mask)
                            
        preds = torch.argmax(logits, dim=-1)
        return logits, preds
