
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):

    def __init__(   
                    self,
                    input_size:int,
                    hidden_size:int, 
                    num_layers:int, 
                    bidirectional:bool, 
                    dropout:float=None,
                    ):
        super().__init__()

        self.lstm = nn.LSTM(     
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers, 
                                bidirectional=bidirectional,  
                                batch_first=True
                            )

        if dropout:
            self.dropout = nn.Dropout(self.DROPOUT)


    def forward(self, X, lengths, padding=0.0):
        
        pass_states = False
        if isinstance(X, tuple):
           #X, h_0, c_0 = X
            X, *states = X
            pass_states = True

        packed_embs = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)

        if pass_states:
            lstm_packed, hidden = self.lstm(packed_embs, states)
        else:
            lstm_packed, hidden = self.lstm(packed_embs)

        unpacked, lengths = pad_packed_sequence(lstm_packed, batch_first=True, padding_value=0.0)

        return unpacked, hidden