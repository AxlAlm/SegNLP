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
    A pointer is learing attention scores for each position over all possible position. Attention scores
    are probility distributions over all possible units in the input.

    These probabilites are interpreted as to where a units it pointing. E.g. if the attention
    scores for a unit at position n is are hightest at position n+2, then we say that
    n points to n+2.


    The works as follows:

    1) we set first decode input cell state and hidden state from encoder outputs

    then for each LSTMCELL timestep we:

    2) apply a ffnn with sigmoid activation

    3) apply dropout

    4) pass output of 3 to lstm cell with reps from prev timestep

    5) apply attention over given decoder at timestep i and all decoder timestep reps

    """


    def __init__(self, input_size:int, hidden_size:int, dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm_cell =  nn.LSTMCell(hidden_size, hidden_size)
        self.attention = CBAttentionLayer(
                                        input_dim=hidden_size,
                                        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size


    def forward(self, input:Tensor, encoder_outputs:Tensor, mask:Tensor, states:Tensor=None):

        seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]
        device = encoder_outputs.device

        if states is None:
            # if no states are given we will initate states
            seq_len = input.shape[1]
            batch_size = input.shape[0]
            device = input.device

            h_s = torch.rand((batch_size, self.hidden_size), device=device)
            c_s = torch.rand((batch_size, self.hidden_size), device=device)
     
        else:
            h_s, c_s = states

            # if states given are bidirectional
            if h_s.shape[-1] == (self.hidden_size/2):
                # We get the last hidden cell states and timesteps and concatenate them for each directions
                # from (NUM_LAYER*DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE) -> (BATCH_SIZE, HIDDEN_SIZE*NR_DIRECTIONS)
                # The cell state and last hidden state is used to start the decoder (first states and hidden of the decoder)
                # -2 will pick the last layer forward and -1 will pick the last layer backwards
                h_s = torch.cat((h_s[-2], h_s[-1]), dim=1)
                c_s = torch.cat((c_s[-2], c_s[-1]), dim=1)
            else:
                raise NotImplementedError()
            
 
        encoder_outputs = self.dropout(encoder_outputs)
        input = self.dropout(input)

        logits = torch.zeros(batch_size, seq_len, seq_len, device=device)
        for i in range(seq_len):

            if i == 0:
                decoder_input = torch.zeros(input[:,0].shape, device=device)
            else:
                decoder_input = input[:,i-1]
            
            decoder_input = torch.sigmoid(self.input_layer(decoder_input))
            h_s, c_s = self.lstm_cell(decoder_input, (h_s, c_s))

            logits[:, i] = self.attention(h_s, encoder_outputs, mask)
        
        preds = torch.argmax(logits, dim=-1)

        return logits, preds
