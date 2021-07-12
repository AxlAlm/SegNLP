



# basics
from typing import List, Tuple, DefaultDict, Dict
from collections import defaultdict
import pandas as pd

# pytorch
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from segnlp import utils


class DirLinkLabeler(nn.Module):


    def __init__(self, 
                input_size:int, 
                hidden_size:int, 
                output_size:int, 
                dropout:float=0.0
                ):
        super().__init__()

        output_size = ((output_size -1) * 2 ) + 1
        self.link_label_clf_layer = nn.Sequential(
                                                    nn.Linear(input_size, hidden_size),
                                                    nn.Tanh(),
                                                    nn.Dropout(dropout),
                                                    nn.Linear(hidden_size, output_size),
                                                )


    # def __get_preds(self, logits: Tensor,  pair_ids:Tensor, directions:Tensor):
        
    #     dir_masks = [
    #                 directions == 0,
    #                 directions == 1,
    #                 directions == 2
    #                 ]

    #     data = defaultdict(lambda:[])
    #     for i in [0,1,2]:
    #         v, l = torch.max(logits[dir_masks[i]], dim=-1)

    #         data["value"].extend(utils.ensure_numpy(v))
    #         data["label_id"].extend(utils.ensure_numpy(l))
    #         data["direction"].extend([i]*len(v))
    #         data["pair_id"].extend(utils.ensure_numpy(pair_ids[dir_masks[i]]))

    #     df = pd.DataFrame(data)
    #     df.sort_values(by=['value'], inplace=True, ascending=False)
    #     best_pairs = df.groupby("pair_id").first()
    #     best_pairs.reset_index(inplace=True)
    #     return best_pairs.to_dict("list")


    def forward(self, input:Tensor): #, pair_ids:Tensor , directions:Tensor):
        logits = self.link_label_clf_layer(input)
        preds = torch.argmax(logits, dim=-1)

        # preds = self.__get_preds(
        #                             logits = logits,
        #                             pair_ids = pair_ids,
        #                             directions = directions,
        #                             )
        return logits, preds
