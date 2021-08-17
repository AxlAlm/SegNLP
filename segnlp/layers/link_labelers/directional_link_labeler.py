



# basics
from typing import List, Tuple, DefaultDict, Dict
from collections import defaultdict
import pandas as pd
import numpy as np

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
                dropout:float=0.0,
                loss_reduction = "mean",
                ignore_index = -1
                ):
        super().__init__()
        self.loss_reduction = loss_reduction
        self.ignore_index = ignore_index

        # if we have link labels {root, POS, NEG} we will predict the folowing labels
        # {None, root, POS, NEG, POS-rev, NEG-rev}
        #  0      1      2    3        4        5 
        # 

        self.link_labels_wo_root = output_size -1
        self.rev_label_treshhold = output_size
        output_size = ((output_size -1) * 2 ) + 2
        self.link_label_clf_layer = nn.Sequential(
                                                    nn.Linear(input_size, hidden_size),
                                                    nn.Tanh(),
                                                    nn.Dropout(dropout),
                                                    nn.Linear(hidden_size, output_size),
                                                )


    def __get_preds(self, 
                    logits: Tensor,  
                    pair_ids:Tensor, 
                    pair_directions:Tensor,
                    pair_sample_ids: Tensor,
                    pair_p1: Tensor,
                    pair_p2: Tensor
                    ):


        
        # dir_masks = [
        #             pair_directions == 0,
        #             pair_directions == 1,
        #             pair_directions == 2
        #             ]
        
        # # for each direction we get the max values and label ids
        # data = defaultdict(lambda:[])
        # for i in [0,1,2]:
        #     v, l = torch.max(logits[dir_masks[i]], dim=-1)

        #     data["value"].extend(utils.ensure_numpy(v))
        #     data["label_id"].extend(utils.ensure_numpy(l))
        #     data["direction"].extend([i]*len(v))
        #     data["pair_id"].extend(utils.ensure_numpy(pair_ids[dir_masks[i]]))
        #     data["sample_id"].extend(utils.ensure_numpy(pair_sample_ids[dir_masks[i]]))
        #     data["p1"].extend(utils.ensure_numpy(pair_p1[dir_masks[i]]))
        #     data["p2"].extend(utils.ensure_numpy(pair_p2[dir_masks[i]]))

        # df = pd.DataFrame(data)
        # df.sort_values(by=['value'], inplace=True, ascending=False)
        # best_pairs = df.groupby("pair_id").first()
        # best_pairs.reset_index(inplace=True)

        #print(best_pairs)

        # we take the max value and labels for each of the Link labels
        v, l = torch.max(logits, dim=-1)

        #then we build a df
        df = pd.DataFrame(
                            {
                            "value": v,
                            "label": l,
                            "direction": pair_directions,
                            "pair_id": pair_ids,
                            "sample_id": pair_sample_ids,
                            "p1": pair_p1,
                            "p2": pair_p2,
                            }
                            )

        # using the df we can sort the values then select the best pair using groupby().first()
        df.sort_values(by=['value'], inplace=True, ascending=False)
        best_pairs = df.groupby("pair_id", sort=False).first()
        best_pairs.sort_values("pair_id", inplace=True)


        best_pairs["label"] = 5

        # filter out all pair where prediction is 0 ( None), which means that the pairs
        # are predicted to not link, (or more correct "there is no relation between")
        no_link_filter = best_pairs["label"] != 0
        best_pairs = best_pairs[no_link_filter]

        # for each pair where the prediction is a X-rev relation we need to swap the order of the 
        # member of the pair. we do this below.
        p1_p2 = best_pairs.loc[:, ["p1", "p2"]].to_numpy()
        p1_p2_new = np.zeros_like(p1_p2)

        rev_preds_filter = (best_pairs["label"] > self.rev_label_treshhold).to_numpy()

        p1_p2_new[rev_preds_filter, 0] = p1_p2[rev_preds_filter, 1]
        p1_p2_new[rev_preds_filter, 1] = p1_p2[rev_preds_filter, 0]
        
        p1_p2_new[~rev_preds_filter, 0] = p1_p2[~rev_preds_filter, 0]
        p1_p2_new[~rev_preds_filter, 1] = p1_p2[~rev_preds_filter, 1]


        print(p1_p2_new)

        best_pairs.loc[:, ["p1", "p2"]]  = p1_p2_new
        
        # as we now have reversed the order of the pairs we need to remove the reverse labels and 
        # select the correct non-"-rev" link_label.
        best_pairs.loc[rev_preds_filter, "label"] -= self.link_labels_wo_root

        best_pairs.groupby("sample_id", sort=False)

        
        print(best_pairs)

        return best_pairs.to_dict("list")


    def forward(self, 
                input:Tensor,
                pair_ids: Tensor,
                pair_directions: Tensor,
                pair_sample_ids: Tensor,
                pair_p1 : Tensor,
                pair_p2 : Tensor,
                ):

        logits = self.link_label_clf_layer(input)
        # preds = torch.argmax(logits, dim=-1)

        link_labels, links = self.__get_preds(
                                    logits = logits,
                                    pair_ids = pair_ids,
                                    pair_directions = pair_directions,
                                    pair_sample_ids = pair_sample_ids,
                                    pair_p1 = pair_p1,
                                    pair_p2 = pair_p2
                                    )

        return logits, preds



    # def loss(self,
    #             logits:Tensor, 
    #             targets:pair_ids, 
    #             false_seg_mask: Tensor, 
    #             false_link_mask: Tensor
    #             ):
    
    #     # we create new targets for each pair. Pairs which are not True pairs are set to
    #     #  XXXX, pairs which 



    #     # then we change

    #     targets = torch.zeros(logits.shape[0])

    #     mask = false_seg_mask + false_link_mask
    #     targets[mask] = 0

    #     #mask_backwards  = 
    #     targets[mask] += nr_link_labels

    #     targets = logits[mask]
    #     targets = 


    #     loss = F.cross_entropy(
    #                             logits, 
    #                             targets, 
    #                             reduction=self.reduction,
    #                             ignore_index=self.ignore_index
    #                             )