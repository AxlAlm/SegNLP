

from nltk import FreqDist
from nltk.corpus import brown as BROWN
from segnlp.resources.stopwords import stopwords

def brown(most_common:int=10000, remove_stopwords:bool=False):
    freq_count = FreqDist(t.lower() for t in BROWN.words() if t not in stopwords)
    return ["<UNK>"] + [t for t,_ in freq_count.most_common(most_common)]