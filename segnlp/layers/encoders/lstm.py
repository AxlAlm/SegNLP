

#basics
from typing import Sequence, Union


#pytorch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch import Tensor


#segnlp 
from segnlp import utils


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
                    weight_init:str = "xavier_uniform",
                    weight_init_kwargs: dict = {}
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

        self.bidir = bidir
        self.hidden_size = hidden_size
        self.output_size = hidden_size * (2 if bidir else 1)

        self.apply(utils.get_weight_init_fn(weight_init, weight_init_kwargs))



    def forward(self, input:Union[Tensor, Sequence[Tensor]], lengths:Tensor, padding_value=0.0):
        
        input_shape = input.shape

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

 
        sorted_lengths, sorted_idxs = torch.sort(lengths, descending=True)
        sorted = torch.equal(sorted_lengths, lengths)

        if not sorted:
            lengths = sorted_lengths
            _ , original_idxs = torch.sort(sorted_idxs, descending=False)
            input = input[sorted_idxs]

        # dropout on input
        input = self.input_dropout(input)

        # if a sample is length == 0, we assume its filled with zeros. So, we remove the sample,
        # and then extend the dims later
        non_zero_lens = lengths != 0
        input = input[non_zero_lens]
        lengths = lengths[non_zero_lens]

        print(input.shape, lengths)

        packed_embs = pack_padded_sequence(input, lengths, batch_first=True)

        #if we are given states this are also passed
        if pass_states:
            lstm_packed, states = self.lstm(packed_embs, (h_0, c_0))
        else:
            lstm_packed, states = self.lstm(packed_embs)

        output, _ = pad_packed_sequence(
                                            lstm_packed, 
                                            batch_first=True, 
                                            padding_value=padding_value,
                                            # total_length = torch.max(lengths)
                                            )

        print("OUTPUT SHAPE", output.shape)

        # if sample lengths are 0, we have removed the samples, hence we add these back
        output_pad = torch.zeros((input_shape[0], input_shape[1], output.size(2)))
        output_pad[non_zero_lens] = output
        output = output_pad

        print("OUTPUT SHAPE 2", output.shape)

        if not sorted:
            output = output[original_idxs]
        
        #dropout on output
        output = self.output_dropout(output)

        return output, states