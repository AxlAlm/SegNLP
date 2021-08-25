

#basics
from typing import List, Union, Dict

from numpy.lib.arraysetops import isin

#segnlp
from .base import Encoder
from segnlp.resources import vocabs


class WordEncoder(Encoder):


    def __init__(self, vocab:Union[list, str]):
        self._name = "words"
        self.pad_value = 0
        self.unk_word = "<UNK>"

        if isinstance(vocab, str):
            self._vocab = getattr(vocabs, vocab)()

        if "<UNK>" != self._vocab[0]:
            assert 'First word in vocab (at index 0) needs to be "<UNK>"'


        self.id2word = dict(enumerate(self._vocab))
        self.word2id = {w:i for i,w in self.id2word.items()}


    def __len__(self):
       return len(self._vocab)
    
    
    @property
    def keys(self):
        return self._vocab


    def encode(self, word):
        return self.word2id.get(word, self.word2id[self.unk_word])


    def decode(self, i):
        return self.id2word[i]