

import torch
import numpy as np


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
    itself (-1) and so on, hence leaving us with a 5 dim one-hot-encodings.
    h = the unit embedding at some position
    
    A sample in out input matrix will look like:

        S = [
            [h0,h1,h2],
            ],

    1) duplicating the matrix at dim==1 nr times equal to max_units ( dim==1)
    
    2) then we reshape the result of 1) so for each unit in each sample is filled with copies of itself.
       this will columns 0 (0:h).
       
    3) then we transpose output of 2) to create columns 1 (h:h+h)
    
    4) then we create the positional one-hot-encodings. These encodings will follow a strict diagonal
       pattern as we can seen in the example above.
       
    5) concatenate output from steps 2,3 and 4, along with the multiplication of output of step 2 and 3 which 
       creates the columns 2 (h*2:(h*2)).

    6)  at this step we have S and can pass it to our linear layer

    7) before we apply softamx we will mask out the pairs that are not possible, e.g. pairs that stand for links
       that span accross the actuall text. Softmax will ignore any -inf value, hence we will change the values for 
       all pairs that are "illegal" to -inf. Then apply softmax

    8) Lastly, we get the predictions by argmaxing.

    
        
        
        after step 6 we get:
           [
                [h0, h0, h0*h0, 0, 0, 1, 0, 0],
                [h0, h1, h0*h1, 0, 0, 0, 1, 0],
                [h0, h2, h0*h2, 0, 0, 0, 0, 1],
                [h1, h0, h1*h0, 0, 1, 0, 0, 0],
                [h1, h1, h1*h1, 0, 0, 1, 0, 0],
                [h1, h2, h1*h2, 0, 0, 0, 1, 0],
                [h2, h0, h2*h0, 1, 0, 0, 0, 0],
                [h2, h1, h2*h1, 0, 1, 0, 0, 0],
                [h2, h2, h2*h2, 0, 0, 1, 0, 0],
            ]
            
            

    """

    def __init__(self, feature_dim, max_units):
        self.max_units = max_units
        self.one_hot_dim = (max_units*2)-1
        input_dim = feature_dim + one_hot_dim
        self.link_clf = torch.nn.Linear(input_dim, 1)


    def forward(self, x):
        max_units = x.shape[1]
        batch_size = x.shape[0]
        shape = (batch_size,max_units,max_units,x.shape[-1])
        
        # step 1 -3
        x = torch.repeat_interleave(x, max_units, dim=1)
        a_m = torch.reshape(x,shape)
        a_m_t = a_m.transpose(2, 1)
        
        # step 4
        one_hots = torch.LongTensor([
                                    np.diag(np.ones(self.one_hot_dim),i)[:max_units,:self.one_hot_dim] 
                                    for i in range(max_units-1, -1, -1)
                                    ])
        one_hots = one_hots.repeat_interleave(batch_size,dim=0)
        one_hots = one_hots.view((batch_size,max_units,max_units,one_hot_dim))
        
        # step 5
        pair_matrix = torch.cat([a_m, a_m_t, a_m*a_m_t, one_hots], axis=-1)
        
        #step 6
        pair_scores = self.link_clf(pair_matrix)

        #pair_matrix = pair_matrix.view((batch_size, max_units*max_units, pair_matrix.shape[-1]))
        
        # step 7
        #MASk ?!?!?
        pair_probs = F.softmax(pair_scores)

        # step 8
        pair_preds = torch.argmax(pair_probs)
    
        return pair_preds