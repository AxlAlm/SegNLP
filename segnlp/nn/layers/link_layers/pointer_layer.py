

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

#segnlp
from segnlp.nn.layers.content_based_attention import CBAttentionLayer


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

    def __init__(self, input_size:int, hidden_size:int, dropout=None):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm_cell =  nn.LSTMCell(input_size, hidden_size)
        self.attention = CBAttentionLayer(
                                        input_dim=hidden_size,
                                        )
        
        self.use_dropout = False
        if dropout:
            self.dropout = nn.Dropout(dropout)
            self.use_dropout = True


    def forward(self, encoder_outputs, encoder_h_s, encoder_c_s, mask, return_softmax=False):

        seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]
        device = encoder_outputs.device

        h_s = encoder_h_s
        c_s = encoder_c_s

        decoder_input = torch.zeros(encoder_h_s.shape, device=device)
        output = torch.zeros(batch_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            
            decoder_input = torch.sigmoid(self.input_layer(decoder_input))

            if self.use_dropout:
                decoder_input = self.dropout(decoder_input)

            h_s, c_s = self.lstm_cell(decoder_input, (h_s, c_s))

            output[:, i] = self.attention(h_s, encoder_outputs, mask, return_softmax=return_softmax)
        
        return output
