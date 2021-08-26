
#basics
from typing import Sequence

# pytorch
import torch.nn.functional as F
import torch
import torch.nn as nn

# segnlp
from segnlp.resources.pos import spacy_pos 
from segnlp.resources.deps import spacy_dep 

class OneHots(nn.Module):

    def __init__(self, labels:list = None):
        self.vocab = dict(enumerate(labels))
        self.output_size = len(self.vocab)
    

    def _encode(self, input:Sequence):
        return [self.vocab[x] for x in input]


    def forward(self, input:Sequence):
        ids = self._encode(input)
        one_hots = F.one_hot(ids, num_classes=self.output_size)
        return one_hots


class PosOneHots(OneHots):

    def __init__(self):
        super().__init__(labels=spacy_pos)


class DepOneHots(OneHots):


    def __init__(self):
        super().__init__(labels=spacy_dep)

 