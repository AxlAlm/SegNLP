
#basics
import string

# pytroch
import torch

# segnlp
from .base import Vocab


class CharVocab(Vocab):

    def _get_vocab(self):
        return  ["*"] + list(string.printable)

    def __getitem__(self, tokens):
        return [torch.LongTensor([self._item2id.get(c, 0) for c in token]) for token in tokens]
        