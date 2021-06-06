

from nltk import FreqDist
from nltk.corpus import brown
from segnlp.resources.stopwords import stopwords


def brown_vocab(most_common:int=10000, remove_stopwords:bool=False):
    freq_count = FreqDist(t.lower() for t in brown.words() if t not in stopwords)
    return [t for t,_ in freq_count.most_common(most_common)]