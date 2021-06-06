

#basics
from typing import List, Union, Dict

#segnlp
from segnlp.resources.vocab import brown_vocab
from segnlp.preprocessing.encoders.base import Encoder


class WordEncoder(Encoder):


    def __init__(self, max_sample_length:int=None, vocab:list=None):
        self._name = "words"
        self._max_sample_length = max_sample_length
        self.pad_value = 0

        if vocab == None:
            self._vocab = brown_vocab()

        self._vocab.append("<UNK>")
        self.id2word = dict(enumerate(self._vocab))
        self.word2id = {w:i for i,w in self.id2word.items()}


    def __len__(self):
       return len(self._vocab)
    
    
    @property
    def keys(self):
        return self._vocab


    def encode(self, word):
        return self.word2id.get(word,self.word2id["<UNK>"])


    def decode(self, i):
        return self.id2word[i]