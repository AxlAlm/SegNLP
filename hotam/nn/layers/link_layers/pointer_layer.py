

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionLayer(nn.Module):

    """
    based on:
    https://arxiv.org/pdf/1612.08994.pdf


    What we are doing:

    We are getting the probabilites across all units in a sample given one of the units. This tells us which 
    unit is important given a unit. We will use this score to treat it as a pointer. .e.g. this units points to 
    the index of the unit where which given the highest score.

    so,

    query is a representations of a unit at position i
    context is representation for all units in the a sample

    1)
    each of these are passed to linear layers so they are trained.

    2)
    then we add the quary to the context. I.e. we add the representation of the unit at i to all units in our sample.

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
        self.Q = nn.Linear(input_dim, input_dim)
        self.C = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1)
        

    def forward(self, query, context, mask=None):

        
        # query = (BATCH_SIZE, 1, INPUT_DIM )
        query_out = self.Q(query.unsqueeze(1))

        # contex_out = (BATCH_SIZE, SEQ_LEN, INPUT_DIM )
        contex_out = self.C(context)
        
        #print(query_out.shape)
        #print(contex_out.shape)
        #print("HELLO", (contex_out + query_out).shape)
        #(BATCH_SIZE, SEQ_LEN)

        u = self.v(torch.tanh(contex_out + query_out)).squeeze(-1)
        
        # masking needs to be done so that the sequence length
        # of each sequence is adhered to. I.e. the non-padded
        # part needs to amount to 1, not the full sequence with padding.
        # apply it like this :
        # http://juditacs.github.io/2018/12/27/masked-attention.html
        
        if mask.dtype != torch.bool:
            mask = mask.type(torch.bool)
        
        u[~mask] = float('-inf')
    
        # (BATCH_SIZE, SEQ_LEN)
        scores = F.softmax(u, dim=-1)

        return scores



class Pointer(nn.Module):

    """
    A pointer is learing attention scores for each position over all possible position. Attention scores
    are probility distributions over all possible units in the input.

    These probabilites are interpreted as to where a units it pointing. E.g. if the attention
    scores for a unit at position n is are hightest at position n+2, then we say that
    n points to n+2.

    """

    def __init__(self, input_size:int, hidden_size:int, dropout=None):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm_cell =  nn.LSTMCell(input_size, hidden_size)
        self.attention = AttentionLayer(
                                        input_dim=hidden_size,
                                        )
        
        self.use_dropout = False
        if dropout:
            self.dropout = nn.Dropout(dropout)
            self.use_dropout = True


    def forward(self, encoder_outputs, encoder_h_s, encoder_c_s, mask):

        seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]
        device = encoder_outputs.device

        #we concatenate the forward direction lstm states
        # from (NUM_LAYER*DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE) -> (BATCH_SIZE, HIDDEN_SIZE*NUM_LAYER)
        layer_dir_idx = list(range(0,encoder_h_s.shape[0],2))
        encoder_h_s = torch.cat([*encoder_h_s[layer_dir_idx]],dim=1)
        encoder_c_s = torch.cat([*encoder_c_s[layer_dir_idx]],dim=1)

        decoder_input = torch.zeros(encoder_h_s.shape, device=device)
        prev_h_s = encoder_h_s
        prev_c_s = encoder_c_s

        #(BATCH_SIZE, SEQ_LEN)
        pointer_probs = torch.zeros(batch_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            
            decoder_input = torch.sigmoid(self.input_layer(decoder_input))

            if self.use_dropout:
                decoder_input = self.dropout(decoder_input)

            prev_h_s, prev_c_s = self.lstm_cell(decoder_input, (prev_h_s, prev_c_s))

            pointer_softmax = self.attention(prev_h_s, encoder_outputs, mask)
        
            pointer_probs[:, i] = pointer_softmax
        
        return pointer_probs
