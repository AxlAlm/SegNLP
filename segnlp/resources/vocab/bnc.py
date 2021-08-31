

#basics
import os
import json

#segnlp
from segnlp.resources.corpus import BNC
from .vocab import Vocab
from segnlp import get_logger


logger = get_logger("BNC VOCAB")


def bnc_vocab(**kwargs):
    
    logger.info("Creating vocab from BNC ...")

    save_path = "/tmp/bnc_word_freqs.pkl"

    if not os.path.exists(save_path):
        
        freqs = dict(BNC().word_freqs())

        with open(save_path, "w") as f:
            json.dump(freqs, f)

    else:
        with open(save_path, "r") as f:
            freqs = json.load(f)

    kwargs["name"] = f"BNC, size = {kwargs['size']}"
    kwargs["freq_dist"] = freqs
    return Vocab(**kwargs)







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
