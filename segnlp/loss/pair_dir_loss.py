

#pytorch 
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairDirLoss(nn.Module):


    def __init__(self, reduction:str="mean", ignore_index:int=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction


    def forward(self,
                logits:Tensor, 
                targets:pair_ids, 
                false_seg_mask: Tensor, 
                false_link_mask: Tensor
                ):
    

        targets = torch.zeros(logits.shape[0])

        mask = false_seg_mask + false_link_mask
        targets[mask] = 0

        #mask_backwards  = 

        targets[mask] += nr_link_labels

        targets = logits[mask]
        targets = 


        loss = F.cross_entropy(
                                logits, 
                                targets, 
                                reduction=self.reduction,
                                ignore_index=self.ignore_index
                                )