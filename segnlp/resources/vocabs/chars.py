
#basics
import string
from dgl import batch

# pytroch
import torch
from torch.nn.utils.rnn import pad_sequence


# segnlp
from .vocab import Vocab

class Char(Vocab):

    def __init__(self):
        super().__init__(vocab = ["*"] + list(string.printable))


    def __getitem__(self, tokens):
        return pad_sequence([torch.LongTensor([self._item2id.get(c, 0) for c in token]) for token in tokens], batch_first  = True)
        