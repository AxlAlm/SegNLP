
import numpy as np

#pytorch
import torch
import torch.nn.functional as F
import torch.nn as nn

#segnlp
from segnlp.nn.utils import pair_matrix


class PairingLayer(torch.nn.Module):

    """

    To predict links between units we will find all possible pairs and estimate the probability that they are linked.
    We do this by creating a matrix where each row in each sample represent a pair of units. A row R is:

       r =  [hi; hj; hi*hj, φ(h, m)] 

    where i, j are indexes of the units, h is the representation of a unit and φ is a one-hot encodings
    representing the relative distance betwen the units i,j, e.g. j-i

    when we have such a matrix we can get the probability that a pair is linked by passing it to a Linear Layer 
    followed by softmax. Then lastly we can get the Maximum Spanning Tree by taking the argmax along axis 2

    To create that matrix following steps will be taken. In this example let us assume that the max 
    units in our datset is 3.
    
    For each sample we want the following matrix:    
    s = [
                [
                [h0, h0, h0*h0, 0, 0, 1, 0, 0],
                [h0, h1, h0*h1, 0, 0, 0, 1, 0], 
                [h0, h2, h0*h2, 0, 0, 0, 0, 1],
                ],
                [
                [h1, h0, h1*h0, 0, 1, 0, 0, 0],
                [h1, h1, h1*h1, 0, 0, 1, 0, 0],
                [h1, h2, h1*h2, 0, 0, 0, 1, 0],
                ],
                [
                [h2, h0, h2*h0, 1, 0, 0, 0, 0],
                [h2, h1, h2*h1, 0, 1, 0, 0, 0],
                [h2, h2, h2*h2, 0, 0, 1, 0, 0],
                ]
            ]
        
    
    NOTE: that each row here is a 1d vector, the rows with h are vector represent a span of values

    THE LAST 5 DIMENSIONS is the one-hot encoding representing the relative position of the paired unit. 5 dimensions 
    is due to the possible relative relations, stretching from ( -(max_units -1), (max_units-1)). In this example
    the max_units==3, which means that a unit can either be related to 2 units behind itself (-2), 1 unit behind
    itself (-1) , 0 units behind itself (0)  and so on, hence leaving us with a 5 dim one-hot-encodings.

    h = unit embedding
    
    A sample in out input matrix will look like:

        S = [
            [h0,h1,h2],
            ],

    1) duplicating the matrix at dim==1 nr times equal to max_units ( dim==1)
    
    2) then we reshape the result of 1) so for each unit in each sample is filled with copies of itself.
       this will columns 0 (0:h).
       
    3) then we transpose output of 2) to create columns 1 (h:h+h)
    
    4) then we create the relattive positional one-hot-encodings. These encodings will follow a strict diagonal
       pattern as we can seen in the example above. 
       
    
    5) concatenate output from steps 2,3 and 4, along with the multiplication of output of step 2 and 3 which 
       creates the columns 2 (h*2:(h*2)).

    6)  at this step we have S and can pass it to our linear layer

    7) lastly, we set the values for all pairs which are not possible and which we dont want to be counter for in our
        loss function (or any softmax, or other activation function) to -inf
    

    """

    def __init__(self, 
                input_dim:int, 
                max_units:int, 
                pair_rep_mode:list=["cat", "multi"], 
                rel_pos:bool=True,
                dropout:float=0.0
                ):
        super().__init__()

        self.pair_rep_mode = pair_rep_mode
        self.rel_pos = rel_pos
        self.max_units = max_units

        self._input_dim = 0

        if rel_pos:
            self._input_dim += (max_units*2)-1

        if "cat" in pair_rep_mode:
            self._input_dim += input_dim*2

        if "multi" in pair_rep_mode:
            self._input_dim += input_dim

        if "sum" in pair_rep_mode:
            self._input_dim += input_dim

        if "mean" in pair_rep_mode:
            self._input_dim += input_dim


        self.link_clf = torch.nn.Linear(self._input_dim, 1)
        torch.nn.init.uniform_(self.link_clf.weight.data,  a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.link_clf.bias.data,  a=-0.05, b=0.05)



    def forward(self, 
                input_tensor:torch.tensor, 
                unit_mask:torch.tensor, 
                ):

        pm = pair_matrix(
                            input_tensor,
                            modes=self.pair_rep_mode,  # we concatenate pairs with the multiplication of members of the pair
                            rel_pos=self.rel_pos,
                            max_units=self.max_units,
                            )

        #step 6
        pair_scores = self.link_clf(pm).squeeze(-1)

        # step 7, for all samples we set the probs for non existing units to inf and the prob for all
        # units pointing to an non existing unit to -inf.
        unit_mask = unit_mask.type(torch.bool)
        pair_scores[~unit_mask]  =  float("-inf")
        pf = torch.flatten(pair_scores, end_dim=-2)
        mf = torch.repeat_interleave(unit_mask, unit_mask.shape[1], dim=0)
        pf[~mf] = float("-inf")
        pair_scores = pf.view(pair_scores.shape)

        return pair_scores