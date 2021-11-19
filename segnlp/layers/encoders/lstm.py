

#basics
from typing import Sequence, Union, Tuple
import numpy as np

#pytorch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.functional import pad
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
                    weight_init : Union[str, dict] = None,
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

        self.bidir = bidir
        self.hidden_size = hidden_size
        self.output_size = hidden_size * (2 if bidir else 1)

        utils.init_weights(self, weight_init)


    def __solve_input(self, input:Union[Tensor, Sequence[Tensor]]) -> Tensor:
        
        h_0, c_0 = None, None,

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
        

        return input, h_0, c_0, pass_states


    def __sort_input(self, input:Tensor, lengths:int) -> Tuple[Tensor, Tensor, Tensor, bool]:
        
        sorted_lengths, sorted_idxs = torch.sort(lengths, descending=True)
        need_sorting = not torch.equal(sorted_lengths, lengths)

        original_idxs = None
        if need_sorting:
            lengths = sorted_lengths
            _ , original_idxs = torch.sort(sorted_idxs, descending=False)
            input = input[sorted_idxs]

        return input, lengths, original_idxs, need_sorting


    def forward(self, input:Union[Tensor, Sequence[Tensor]], lengths:Tensor, padding_value=0.0) -> Tuple[Tensor,Tuple[Tensor,Tensor]]:

        # fix the input 
        input, h_0,c_0, pass_states =  self.__solve_input(input)

        # sort the input tensors by lengths
        input, lengths, original_idxs, sorting_done = self.__sort_input(input, lengths)

        # if a sample is length == 0, we assume its filled with zeros. So, we remove the sample,
        # and then extend the dims later
        non_zero_lens = lengths != 0
        
        # remove paddings and pack it, turn to 2d
        packed_embs = pack_padded_sequence(
                                            input[non_zero_lens], 
                                            utils.ensure_numpy(lengths[non_zero_lens]),
                                            batch_first=True
                                            )
        
        #if we are given states this are also passed
        if pass_states:
            lstm_packed, states = self.lstm(packed_embs, (h_0, c_0))
        else:
            lstm_packed, states = self.lstm(packed_embs)

        # return to 3D and pad
        output, _ = pad_packed_sequence(
                                            lstm_packed, 
                                            batch_first=True, 
                                            padding_value=padding_value,
                                            )


        # If we had sample of 0 length, we pad outputs so we are back to the original shape
        # we pad on the ends of dim 0 and dim 1
        output = pad(   
                    output, 
                    (0, 0, 0, abs(output.size(1) - input.size(1)), 0, abs(output.size(0) - input.size(0))), 
                    mode = 'constant', 
                    value = 0
                    )


        if sorting_done:
            output = output[original_idxs]
   

        return output, states