
import numpy as np

#pytorch
import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn

#segnlp
from segnlp.layers.general import LinearCLF


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

    6)  at this step we have S and can pass it to our linear layer

    7) lastly, we set the values for all pairs which are not possible and which we dont want to be counted for in our
        loss function  to -inf. All pairs across padded segments are set to -inf

    """

    def __init__(self, 
                input_size:int, 
                max_segments:int, 
                mode:list=["cat", "multi"], 
                rel_pos:bool=True,
                ):
        super().__init__()

        self.mode = mode
        self.rel_pos = rel_pos
        self.max_segments = max_segments

        self._input_size = 0

        if rel_pos:
            self._input_size += (max_segments*2)-1

        if "cat" in mode:
            self._input_size += input_size*2

        if "multi" in mode:
            self._input_size += input_size

        if "sum" in mode:
            self._input_size += input_size

        if "mean" in mode:
            self._input_size += input_size

        self.link_clf = LinearCLF(
                                    input_size = self._input_size, 
                                    output_size = 1
                                    )


    def __create_matrix(self, input:Tensor) -> Tensor:        
        device = input.device
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
        

        #adding one_hot encoding for the relative position
        if self.rel_pos:
            one_hot_dim = (self.max_segments*2)-1
            one_hots = torch.tensor(
                                        [
                                        np.diag(np.ones(one_hot_dim),i)[:dim1,:one_hot_dim] 
                                        for i in range(dim1-1, -1, -1)
                                        ], 
                                        dtype=torch.uint8,
                                        device=device
                                        )
            one_hots = one_hots.repeat_interleave(batch_size, dim=0)
            one_hots = one_hots.view((batch_size, dim1, dim1, one_hot_dim))
            
            to_cat.append(one_hots)

        pair_matrix = torch.cat(to_cat, axis=-1)
        
        return pair_matrix


    def forward(self, 
                input_tensor:torch.tensor, 
                segment_mask:torch.tensor, 
                ):

        pm = self.__create_matrix(input_tensor)

        #predict links
        pair_logits, _ = self.link_clf(pm)
        pair_logits = pair_logits.squeeze(-1)

        # for all samples we set the probs for non existing segments to inf and the prob for all
        # segments pointing to an non existing segment to -inf.
        segment_mask = segment_mask.type(torch.bool)
        pair_logits[~segment_mask]  =  float("-inf")
        pf = torch.flatten(pair_logits, end_dim=-2)
        mf = torch.repeat_interleave(segment_mask, segment_mask.shape[1], dim=0)
        pf[~mf] = float("-inf")
        logits = pf.view(pair_logits.shape)
        preds = torch.argmax(logits, dim=-1)

        return logits, preds 
