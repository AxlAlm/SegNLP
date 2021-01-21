

from hotam.preprocessing.encoders.base import Encoder
#from nltk.corpus import words
from ...resources.vocab import vocab
from typing import List, Union, Dict
#import json

class WordEncoder(Encoder):


    def __init__(self, max_sample_length:int=None):
        self._name = "words"
        self._max_sample_length = max_sample_length
        self.pad_value = 0
        self.word2id = vocab
        self.id2word = {i:w for w,i in self.word2id.items()}
        self.unkown = "<unkown>"


    def __len__(self):
       len(self._vocab)
    
    
    @property
    def keys(self):
        return self._vocab


    def encode(self, word):
        return self.word2id.get(word,self.word2id["<unkown>"])


    def decode(self, i):
        return self.id2word[i]