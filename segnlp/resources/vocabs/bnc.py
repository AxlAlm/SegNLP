

#basics
import os
import json
from nltk import FreqDist

#segnlp
from segnlp.resources.corpus import BNC
from .vocab import Vocab
from segnlp import get_logger


logger = get_logger("BNC VOCAB")


def bnc_vocab(
        size: int = 30000, 
        unk_word: str = "<UNK>",
        remove_stopwords:bool = False
        ) -> Vocab:
    
    save_path = "/tmp/bnc_word_freqs.json"

    if not os.path.exists(save_path):
        
        freqs = BNC().word_freqs()

        with open(save_path, "w") as f:
            json.dump(freqs, f)

    else:
        with open(save_path, "r") as f:
            freqs = json.load(f)


    return Vocab(
        name = "BNC", 
        freq_dist = freqs, 
        size = size, 
        unk_word = unk_word,
        remove_stopwords = remove_stopwords,
    )







# def bnc(most_common:int=10000, remove_stopwords=False):
    
#     save_path = "/tmp/bnc_vocab.pkl"
#     if os.path.exists(save_path):

#         with open(save_path, "rb") as f:
#             vocab = pkl.load(f)
        
#     else:

#         freq_count = Counter()
#         for doc in BNC():
#             freq_count += Counter(doc.lower().split())

#         if remove_stopwords:
#             for sw in stopwords:
#                 if sw in freq_count:
#                     del freq_count[sw]

#         vocab = ["<UNK>"] + [t for t,_ in freq_count.most_common(most_common)]

#         with open(save_path, "wb") as f:
#             pkl.dump(vocab, f)

#     return vocab
