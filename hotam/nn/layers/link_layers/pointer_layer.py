



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




class Decoder(nn.Module):


    def __init__(self, input_size:int, hidden_size:int, dropout=None):
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm_cell =  nn.LSTMCell(input_size, hidden_size)

        self.attention = CONTENT_BASED_ATTENTION(
                                                    input_dim=hidden_size,
                                                    )
        
        self.use_dropout = False
        if dropout:
            self.dropout = nn.Dropout(dropout)
            self.use_dropout = True


    def forward(self, encoder_outputs, encoder_h_s, encoder_c_s, mask):

        seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]

        #we concatenate the forward direction lstm states
        # from (NUM_LAYER*DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE) ->
        # (BATCH_SIZE, HIDDEN_SIZE*NUM_LAYER)
        layer_dir_idx = list(range(0,encoder_h_s.shape[0],2))
        encoder_h_s = torch.cat([*encoder_h_s[layer_dir_idx]],dim=1)
        encoder_c_s = torch.cat([*encoder_c_s[layer_dir_idx]],dim=1)

        decoder_input = torch.zeros(encoder_h_s.shape)
        prev_h_s = encoder_h_s
        prev_c_s = encoder_c_s

        #(BATCH_SIZE, SEQ_LEN)
        pointer_probs = torch.zeros(batch_size, seq_len, seq_len)
        for i in range(seq_len):
            
            prev_h_s, prev_c_s = self.lstm_cell(decoder_input, (prev_h_s, prev_c_s))

            if self.use_dropout:
                prev_h_s = self.dropout(prev_h_s)

            decoder_input = self.input_layer(decoder_input)
            decoder_input = F.sigmoid(decoder_input)

            pointer_softmax = self.attention(prev_h_s, encoder_outputs, mask)
        
            pointer_probs[:, i] = pointer_softmax
            
        return pointer_probs