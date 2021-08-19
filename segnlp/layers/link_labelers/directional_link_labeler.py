



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


    """
    Direktional Link Labeler work on the level of all possible pairs. It takes all bi-directional pairs
    between segments and predicts the directional labels including None for no link and root for pointing 
    to itself.
    
    
    """


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

        # if we have link labels {root, REL1, REL2} we will predict the folowing labels
        # {None, root, REL1, REL2, REL1-rev, REL2-rev}
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
                    pair_p1: Tensor,
                    pair_p2: Tensor
                    ):

        # we take the max value and labels for each of the Link labels
        v, l = torch.max(logits, dim=-1)

        #then we build a df
        df = pd.DataFrame(
                            {
                            "value": v.detach().numpy(),
                            "label": l.detach().numpy(),
                            "p1": pair_p1,
                            "p2": pair_p2,
                            }
                            )

        # filter out all pair where prediction is 0 ( None), which means that the pairs
        # are predicted to not link, (or more correct "there is no relation between")
        df = df[df["label"] != 0]

        # for each pair where the prediction is a X-rev relation we need to swap the order of the 
        # member of the pair. I.e. if we have a pair ( p1, p2) where the label is X-rev we swap places
        # on p1 and p2. What this does is make p1 our column for SOURCE and p2 our column for TARGET. 
        # this means that the value of p2 is the index of all the segments in a sample p1 is related to.
        p1_p2 = df.loc[:, ["p1", "p2"]].to_numpy()
        p1_p2_new = np.zeros_like(p1_p2)

        rev_preds_filter = (df["label"] > self.rev_label_treshhold).to_numpy()

        p1_p2_new[rev_preds_filter, 0] = p1_p2[rev_preds_filter, 1]
        p1_p2_new[rev_preds_filter, 1] = p1_p2[rev_preds_filter, 0]
        
        p1_p2_new[~rev_preds_filter, 0] = p1_p2[~rev_preds_filter, 0]
        p1_p2_new[~rev_preds_filter, 1] = p1_p2[~rev_preds_filter, 1]

        df.loc[:, ["p1", "p2"]]  = p1_p2_new

        #after we have set SOURCE and TARGETS we normalize the link_labels to non-"-Rev" labels
        df.loc[rev_preds_filter, "label"] -= self.link_labels_wo_root

        #as we also removed 0 labels above we will move all labels down 1 so we get original labels
        df.loc[:, "label"] -= 1

        # Lastly we want to sort all the pairs then select the row for
        # for each unique p1, starting segments. I.e. we get the highested scored
        # link and link_label for any unqiue segment p1
        df.sort_values(by=['value'], inplace=True, ascending=False)
        seg_df = df.groupby("p1").first()

        link_label_preds = seg_df["label"].to_numpy()
        links = seg_df["p2"].to_numpy()

        return link_label_preds, links


    def forward(self, 
                input:Tensor,
                pair_p1 : Tensor,
                pair_p2 : Tensor,
                ):

        logits = self.link_label_clf_layer(input)
        link_labels, links = self.__get_preds(
                                    logits = logits,
                                    pair_p1 = pair_p1,
                                    pair_p2 = pair_p2
                                    )

        return logits, (
                        {
                        "preds":link_labels,
                        "level": "p_seg",
                        "task": "link_label"
                        },
                        {
                        "preds": links,
                        "level": "p_seg",
                        "task": "link"
                        }
                        )


    def loss(self,
            targets: Tensor, 
            logits: Tensor, 
            directions: Tensor,
            neg_mask: Tensor, 
            ):

        # target labels are not directional so we need to make them so. Targets also lack
        # label for no link. So, we first add 1 too all labels moving them up freeing 0 for None label for no links.
        # then for all directions which are 2 (backwards/reversed) we add the number of link_labels (minus root)
        targets += 1
        targets[directions == 2] += self.link_labels_wo_root
    
        # neg_mask include all the pairs which should be countet as non linking pairs, e.g. label None.
        # this mask is usually created from all segs that are not linked or pairs that include segments which 
        # are not true segments.
        targets[neg_mask] = 0

        loss = F.cross_entropy(
                                logits, 
                                targets, 
                                reduction=self.loss_reduction,
                                ignore_index=self.ignore_index
                                )

        return loss