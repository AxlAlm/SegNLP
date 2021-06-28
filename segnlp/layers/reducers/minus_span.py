
#pytorch
import torch
import torch.nn as nn
from torch import Tensor


class MinusSpan(nn.Module):


    def __init__(self, input_size:int, dropout:float=0.0, fine_tune:bool=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.output_size = hidden_dim*4

        self.dropout = nn.Dropout(dropout)

        self.fine_tune = fine_tune
        if fine_tune:
            self.ft = nn.Linear(self.output_size, self.output_size)


    def forward(self, input:Tensor, span_idxs:Tensor):

        """
        based on paper:
        https://www.aclweb.org/anthology/P19-1464.pdf
        (above paper refers to: https://www.aclweb.org/anthology/P16-1218.pdf)

        Minus based representations are away to represent segments given a representations
        of the segment as a set of e.g. tokens. We subtract LSTM hidden vectors to create a 
        vector representation of a segment.

        NOTE! in https://www.aclweb.org/anthology/P19-1464.pdf:
        "Each ADU span is denoted as (i, j), where i and j are word indices (1 ≤ i ≤ j ≤ T)."
        but for to make the spans more pythonic we define them as  (0 ≤ i ≤ j < T)

        (NOTE!  φ(wi:j), whihc is included in the forumla i the paper,  is added later because of Distiction between AM and AC)

        minus span reps are following:
            h(i,j) = [
                        →hj − →hi-1;
                        ←hi − ←hj+1; 
                        →hi-1;← hj+1    
                        ]

        NOTE! we assume that ←h is aligned with →h, i.e.  ←h[i] == →h[i] (the same word)
        
        So, if we have the following hidden outputs; 
                                                   _______span________
            fh (forward_hidden)   = [word0, word1, word2, word3, word4, word5]
            bh (backwards hidden) = [word0, word1, word2, word3, word4, word5] 
                                                   _______span________
        
        h(2,4) = [
                    word4 - word1;
                    word2 - word5;
                    word1,
                    word5 
                    ]

        So, essentially we take previous and next segments from and subtract them from the current segment to create 
        a representation that include information about previous and next segments :D

        """
        batch_size, nr_seq, _ = input.shape

        input = self.dropout(input)

        forward = input[: , :, :self.input_size]
        backward = input[: , :, self.input_size:]

        minus_reps = torch.zeros((batch_size, nr_seq, self.output_size), device=self.device)
        
        for k in range(batch_size):
            for n,(i,j) in enumerate(span_idxs[k]):

                if i==0 and j == 0:
                    continue

                if i-1 == -1:
                    f_pre = torch.zeros(self.hidden_dim, device=self.device)
                else:
                    f_pre = forward[k][i-1]

                if j+1 >= backward.shape[1]:
                    b_post = torch.zeros(self.hidden_dim,  device=self.device)
                else:
                    b_post = backward[k][j+1]

                f_end = forward[k][j]
                b_start = backward[k][i]

                minus_reps[k][n] = torch.cat((
                                                    f_end - f_pre,
                                                    b_start - b_post,
                                                    f_pre, 
                                                    b_post
                                                    ))

        if self.fine_tune:
            minus_reps = self.ft(minus_reps)

        return minus_reps

