
#basics
from typing import List, Iterator, Union
from collections import Counter

#NLTK
from nltk import FreqDist

#segnlp 
from segnlp.resources.stopwords import stopwords


class Vocab:

    def __init__(self, 
                freq_dist: Union[FreqDist, dict], 
                size: int = 30000, 
                unk_word: str = "<UNK>",
                remove_stopwords:bool=False
                ):
        self.unk_word = unk_word
        self._vocab = FreqDist(freq_dist) if isinstance(freq_dist, dict) else freq_dist

        #remove stopwords
        if remove_stopwords:
            for sw in stopwords:
                del self._vocab[sw]

        #set vocabulary size, remove least common words
        if size is not None:
            self._vocab = list(dict(self._vocab.most_common(size)).keys())
        else:
            self._vocab = list(dict(self._vocab).keys())

        self._id2word = dict(enumerate([self.unk_word] + self._vocab))
        self._word2id = {w:i for i,w in self._id2word.items()} 


    @classmethod
    def from_corpus(self, 
                    corpus : Iterator,
                    size: int = 30000,
                    unk_str: str = "<UNK>",
                    remove_stopwords:bool=False
                    ):
        Vocab(
                freq_dist = FreqDist(corpus),
                size = size,
                unk_str = unk_str,
                remove_stopwords = remove_stopwords,
                )

    def __len__(self):
       return self.size


    def __getitem__(self, words: List[str]):
        return [self._word2id.get(word, 0) for word in words]


    def decode(self, word_ids: List[str]):
        return [self._id2word[i] for i in  word_ids]