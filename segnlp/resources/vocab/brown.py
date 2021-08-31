

#nltk
from nltk.corpus import brown as BROWN

#segnlp
from .vocab import Vocab

def brown_vocab(**kwargs):
    kwargs["name"] = f"brown, size = {kwargs['size']}"
    kwargs["corpus"] = (t.lower() for t in BROWN.words())
    return Vocab.from_corpus(**kwargs)
   
    
