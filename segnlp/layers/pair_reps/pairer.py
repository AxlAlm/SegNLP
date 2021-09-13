
#basics
from typing import Union
import numpy as np

#pytorch
import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn


class Pairer(torch.nn.Module):

    """

    To predict links between segments we will find all possible pairs and estimate the probability that they are linked.
    We do this by creating a matrix where each row in each sample represent a pair of segments. Let say we want to represent a pair
    the embedding of the first and second member, as well as the dot product of the member and lastly a positional encoding. Then
    a row would be 

       r =  [hi; hj; hi*hj, φ(h, m)] 

    where i, j are indexes of the segments, h is the representation of a segment and φ is a one-hot encodings
    representing the relative distance betwen the segments i,j (j-i)

    when we have such a matrix we score the all the pairs and select the pair which scores highest using Maximum Spanning Tree.
    I.e. simply taking the argmax along axis 2.

    To create that matrix following steps will be taken. In this example let us assume that the max 
    segments in our datset is 3.
    
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

    THE LAST 5 DIMENSIONS is the one-hot encoding representing the relative position of the paired segment. 5 dimensions 
    is due to the possible relative relations, stretching from ( -(max_segments -1), (max_segments-1)). In this example
    the max_segments==3, which means that a segment can either be related to 2 segments behind itself (-2), 1 segment behind
    itself (-1) , 0 segments behind itself (0)  and so on, hence leaving us with a 5 dim one-hot-encodings.

    h = segment embedding
    
    A sample in out input matrix will look like:

        S = [
            [h0,h1,h2],
            ],

    1) duplicating the matrix at dim==1 nr times equal to max_segments ( dim==1)
    
    2) then we reshape the result of 1) so for each segment in each sample is filled with copies of itself.
       this will columns 0 (0:h).
       
    3) then we transpose output of 2) to create columns 1 (h:h+h)
    
    4) then we create the relattive positional one-hot-encodings. These encodings will follow a strict diagonal
       pattern as we can seen in the example above. 
       
    5) concatenate output from steps 2,3 and 4, along with the multiplication of output of step 2 and 3 which 
       creates the columns 2 (h*2:(h*2)).
    """

    def __init__(self, 
                input_size:int, 
                mode:list=["cat", "multi"], 
                n_rel_pos:int=None,
                ):
        super().__init__()

        self.mode = mode
        self.n_rel_pos = n_rel_pos

        self.output_size = 0
        if n_rel_pos:
            self.output_size += (self.n_rel_pos*2)-1

        if "cat" in mode:
            self.output_size += input_size*2

        if "multi" in mode:
            self.output_size += input_size

        if "sum" in mode:
            self.output_size += input_size

        if "mean" in mode:
            self.output_size += input_size


    def forward(self,
                input:Tensor,
                device : Union[str, torch.device] = "cpu",
                ) -> Tensor:    

        batch_size = input.shape[0]
        dim1 = input.shape[1]

        shape = (batch_size, dim1, dim1, input.shape[-1])
        m = torch.reshape(torch.repeat_interleave(input, dim1, dim=1), shape)
        mT = m.transpose(2, 1)

        to_cat = []
        if "cat" in self.mode:
            to_cat.append(m)
            to_cat.append(mT)
        
        if "multi" in self.mode:
            to_cat.append(m*mT)

        if "mean" in self.mode:
            to_cat.append((m+mT /2))

        if "sum" in self.mode:
            to_cat.append(m+mT)
        

        #adding one hot encodings for the relative position
        if self.n_rel_pos:
            rel_one_hot_dim = (self.n_rel_pos*2)-1
            rel_one_hots = torch.tensor(
                                        [
                                        np.diag(np.ones(rel_one_hot_dim),i)[:dim1,:rel_one_hot_dim] 
                                        for i in range(dim1-1, -1, -1)
                                        ], 
                                        dtype=torch.uint8,
                                        device=device
                                        )
            rel_one_hots = rel_one_hots.repeat_interleave(batch_size, dim=0)
            rel_one_hots = rel_one_hots.view((batch_size, dim1, dim1, rel_one_hot_dim))
            
            to_cat.append(rel_one_hots)

        pair_matrix = torch.cat(to_cat, axis=-1)
        
        return pair_matrix

