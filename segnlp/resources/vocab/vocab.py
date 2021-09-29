
#basics
from typing import List, Iterator, Union
from collections import Counter

#NLTK
from nltk import FreqDist

#pytorch
import torch
from torch._C import dtype

#segnlp 
from segnlp.resources.stopwords import stopwords


class Vocab:

    def __init__(self, 
                name: str, 
                freq_dist: Union[FreqDist, dict], 
                size: int = 30000, 
                unk_word: str = "<UNK>",
                remove_stopwords:bool = False,
                #return_as_tensor: bool = True
                ):
        #self.return_as_tensor = True
        self._name = name
        self.unk_word = unk_word
        self._fred_dist = FreqDist(freq_dist) if isinstance(freq_dist, dict) else freq_dist

        #remove stopwords
        if remove_stopwords:
            for sw in stopwords:
                del self._fred_dist[sw]

        #set vocabulary size, remove least common words.
        # always +1 for UNK
        if size is not None:
            self._vocab = [self.unk_word] + list(dict(self._fred_dist.most_common(size - 1)).keys())
        else:
            self._vocab = [self.unk_word] + list(dict(self._fred_dist).keys())

        self._id2word = dict(enumerate(self._vocab))
        self._word2id = {w:i for i,w in self._id2word.items()} 

        self._size = len(self._vocab)


    # @classmethod
    # def from_corpus(self, 
    #                 corpus : Iterator,
    #                 size: int = 30000,
    #                 unk_str: str = "<UNK>",
    #                 remove_stopwords:bool=False
    #                 ):
    #     Vocab(
    #             freq_dist = FreqDist(corpus),
    #             size = size,
    #             unk_str = unk_str,
    #             remove_stopwords = remove_stopwords,
    #             )

    @property
    def vocab(self):
        return self._id2word

    @property
    def name(self):
        return self._name

    def __str__(self) -> str:
        return self._name

    def __len__(self):
        return self._size
    

    def __getitem__(self, words: Union[str, List[str]]):

        if isinstance(words, str):
            words = [words]
            
        return torch.tensor([self._word2id.get(word, 0) for word in words], dtype=torch.int64)


    # def decode(self, word_ids: Union[str, List[str]]):

    #     return [self._id2word[i] for i in  word_ids]