

#pytorch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor


class LSTM(nn.Module):

    def __init__(   
                    self,
                    input_size:int,
                    hidden_size:int, 
                    num_layers:int, 
                    bidir:bool, 
                    dropout:float=0.0,
                    w_init:str="xavier_uniform"
                    ):
        super().__init__()

        self.lstm = nn.LSTM(     
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers, 
                                bidirectional=bidir,  
                                batch_first=True
                            )
        self.dropout = nn.Dropout(dropout)
        self.__initialize_weights(w_init)
        self.output_size = hidden_size * (2 if bidir else 1)

    

    def __initialize_weights(self, w_init:str):
        for name, param in self.lstm.named_parameters():
            
            if 'bias' in name:
                nn.init.constant(param, 0.0)

            elif 'weight' in name:

                if w_init == "orthogonal":
                    nn.init.orthogonal_(param)

                elif w_init == "xavier_uniform":
                    nn.init.xavier_uniform_(param)

                else:
                    raise RuntimeError()


    def forward(self, input:Tensor, lengths:Tensor, padding=0.0):

        input = self.dropout(input)
        
        pass_states = False
        if isinstance(input, tuple):
           #X, h_0, c_0 = X
            input, *states = input
            pass_states = True

        packed_embs = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)

        if pass_states:
            lstm_packed, hidden = self.lstm(packed_embs, states)
        else:
            lstm_packed, hidden = self.lstm(packed_embs)

        unpacked, lengths = pad_packed_sequence(lstm_packed, batch_first=True, padding_value=0.0)

        return unpacked, hidden