

#basics
from collections import Counter

#segnlp
from segnlp.resources.corpus import BNC
from segnlp.resources.stopwords import stopwords

def bnc(most_common:int=10000, remove_stopwords=False):

    freq_count = Counter()
    for doc in BNC():
        freq_count += Counter(doc.lower().split())

    if remove_stopwords:
        for sw in stopwords:
            if sw in freq_count:
                del freq_count[sw]

    return ["<UNK>"] + [t for t,_ in freq_count.most_common(most_common)]
