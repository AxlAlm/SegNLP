
#basics
import string

# pytroch
import torch

# segnlp
from .vocab import Vocab


class Char(Vocab):

    def __init__(self):
        super().__init__(vocab = ["*"] + list(string.printable))


    def __getitem__(self, tokens):
        return [torch.LongTensor([self._item2id.get(c, 0) for c in token]) for token in tokens]
        