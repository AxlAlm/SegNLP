

#pytorch
from typing import Sequence, Union
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch import Tensor


class LSTM(nn.Module):

    def __init__(   
                    self,
                    input_size:int,
                    hidden_size:int, 
                    num_layers:int, 
                    bidir:bool, 
                    dropout:float=0.0,
                    input_dropout:float=0.0,
                    output_dropout:float=0.0,
                    #reproject:str = None, #linear reprojection of input to the hidden size dim
                    w_init:str="xavier_uniform",
                    sorted:bool = True,
                    ):
        super().__init__()


        self.lstm = nn.LSTM(     
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers, 
                                bidirectional=bidir,  
                                batch_first=True,
                                dropout = dropout
                            )
        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        self.__initialize_weights(w_init)
        self.bidir = bidir
        self.hidden_size = hidden_size
        self.output_size = hidden_size * (2 if bidir else 1)
        self.sorted = sorted
    

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


    def forward(self, input:Union[Tensor, Sequence[Tensor]], lengths:Tensor, padding_value=0.0):
        
        # if input in a sequence we concatentate the tensors
        # if the second input element is a tuple its assumed its the states (h0,c0)
        pass_states = False
        if not isinstance(input, Tensor):
                
            #to take care of given states
            if isinstance(input[1], tuple):
                input, (h_0, c_0) = input

                # If states are bidirectional and the LSTM is not. The hidden dim of the states needs to be
                # 1/2 of the LSTMs hidden dim. The states will be concatenated in terms of direction
                # and passed as states to the LSTM
                # from (NUM_LAYER*DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE) -> (1, BATCH_SIZE, HIDDEN_SIZE*NR_DIRECTIONS)
                if h_0.shape[-1] == (self.hidden_size/2) and not self.bidir:

                    # The cell state and last hidden state is used to start the decoder (first states and hidden of the decoder)
                    # -2 will pick the last layer forward and -1 will pick the last layer backwards
                    h_0 = torch.cat((h_0[-2], h_0[-1]), dim=1).unsqueeze(0)
                    c_0 = torch.cat((c_0[-2], c_0[-1]), dim=1).unsqueeze(0)
           
                pass_states = True
        
            else:
                input = torch.cat(input, dim = -1)


        if not self.sorted:
            sorted, sorted_idxs = torch.sort(lengths, descending=True)
            _ , original_idxs = torch.sort(sorted_idxs, descending=False)
            input = input[sorted_idxs]

        # dropout on input
        input = self.input_dropout(input)
        

        packed_embs = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)

        #if we are given states this are also passed
        if pass_states:
            lstm_packed, states = self.lstm(packed_embs, (h_0, c_0))
        else:
            lstm_packed, states = self.lstm(packed_embs)

        output, lengths = pad_packed_sequence(lstm_packed, batch_first=True, padding_value=padding_value)

        if not self.sorted:
            output = output[original_idxs]
        
        #dropout on output
        output = self.output_dropout(output)

        return output, states