

#basics
from collections import Counter
import os
import pickle as pkl

#segnlp
from segnlp.resources.corpus import BNC
from segnlp.resources.stopwords import stopwords

def bnc(most_common:int=10000, remove_stopwords=False):
    
    save_path = "/tmp/bnc_vocab.pkl"
    if os.path.exists(save_path):

        with open(save_path, "rb") as f:
            vocab = pkl.load(f)
        
    else:

        freq_count = Counter()
        for doc in BNC():
            freq_count += Counter(doc.lower().split())

        if remove_stopwords:
            for sw in stopwords:
                if sw in freq_count:
                    del freq_count[sw]

        vocab = ["<UNK>"] + [t for t,_ in freq_count.most_common(most_common)]

        with open(save_path, "wb") as f:
            pkl.dump(vocab, f)

    return vocab