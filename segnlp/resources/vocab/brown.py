

#nltk
from nltk.corpus import brown as BROWN

#segnlp
from .vocab import Vocab

def brown_vocab(
                size: int = 30000, 
                unk_word: str = "<UNK>",
                remove_stopwords:bool = False,
                )-> Vocab:

    return Vocab(
                name = "Brown",
                freq_dist = FreqDist((t.lower() for t in BROWN.words())),
                size = size,
                unk_str = unk_word,
                remove_stopwords = remove_stopwords,
                )
