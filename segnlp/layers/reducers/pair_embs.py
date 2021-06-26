        
        
        
#pytroch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from segnlp import utils

class PairEmb(nn.Module):

    def __init__(self):
        super().__init__()
        

    def forward(self, token_embs: Tensor, pair_data:dict) -> Tensor:
        
        # Indexing sequential lstm hidden state using two lists of
        # start and end ids of each unit
        # from list(tuple(start_p1, start_p2)) to two separate lists
        # NOTE We have used the function to split the list of list of tuples
        # `split_nested_list` four times.
        p1_st, p2_st = self.split_nested_list(pair_data["start"], self.device)
        p1_end, p2_end = self.split_nested_list(pair_data["end"], self.device)
        p1_s = utils.range_3d_tensor_index(token_embs,
                                     start=p1_st,
                                     end=p1_end,
                                     pair_batch_num=pair_data['lengths'],
                                     reduce_="mean")
        p2_s = utils.range_3d_tensor_index(token_embs,
                                     start=p2_st,
                                     end=p2_end,
                                     pair_batch_num=pair_data['lengths'],
                                     reduce_="mean")
        
        pair_embs = torch.cat((p1_s, p2_s), dim=-1)
        
        return pair_embs