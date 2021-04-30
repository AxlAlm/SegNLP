

import string
from nltk.corpus import words
from segnlp.preprocessing.encoders.base import Encoder
from typing import List, Union, Dict
import numpy as np


class CharEncoder(Encoder):
    """ Encoder for characters"""

    def __init__(self, max_sample_length:int=None):
        self._name = "chars"
        self.id2char = dict(enumerate(string.printable, start=1))
        self.char2id = {c:i for i,c in self.id2char.items()}
        self._max_sample_length = max_sample_length
        self.max_word_length = max(map(len,words.words()))
        self.pad_value = 0
        self.unkown = "*"

    def __len__(self):
        return len(self.id2char)

    @property
    def keys(self):
        return list(char2id.keys())    


    def encode(self, word:str) -> List[int]:
        """given a word encodes the characters 

        Parameters
        ----------
        word : str
            word to encode
        pad : bool, optional
            if pad or not, by default False

        Returns
        -------
        List[int]
            list of int for encoded characters
        """

        #if pad:
        padded = np.zeros((self.max_word_length,))
        #padded.fill(self.pad_value)
        for i, c in enumerate(word):
            padded[i] = self.char2id.get(c, self.char2id["*"])
        return padded.tolist()
        #else:
            #return np.array([self.char2id.get(c, self.char2id["*"]) for c in word])


    def decode(self,char_ids:List[int]) -> str:
        """given list of character encoding creates a word

        Parameters
        ----------
        char_ids : List[int]
            list of character ids for encoded words

        Returns
        -------
        str
            decoded word
        """
        return "".join([self.id2char[i] for i in char_ids if i != self.pad_value])


    def encode_list(self, word_list:List[str], pad=False) -> List[List[int]]:
        """List of words to decode 

        Parameters
        ----------
        word_list : List[str]
            list of words to encode
        pad : bool, optional
            pad or not, by default False

        Returns
        -------
        List[List[int]]
            list of list of character encodings 
        """
        # if pad:
        #     padded = np.zeros((self.max_sample_length, self.max_word_length))
        #     padded.fill(self.pad_value)
        #     for i, word in enumerate(word_list):
        #         padded[i] = self.encode(word,pad=pad)
        #     return padded
        # else:
        return np.array([self.encode(word) for word in word_list])
