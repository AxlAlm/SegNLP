
#basics
import pathlib
import os


# segnlp
from segnlp.utils import load_json
from .vocab import Vocab


class BNC_10k(Vocab):

    def __init__(self):
        fp = os.path.join(pathlib.Path(__file__).parent.parent, "freqs", "bnc_10k.json")
        freqs = load_json(fp)
        vocab =  list(freqs.keys())
        super().__init__(vocab = vocab)