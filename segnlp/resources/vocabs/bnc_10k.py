
#basics
import pathlib
import os


# segnlp
from segnlp.utils import load_json
from .base import Vocab


class BNC_10k(Vocab):

    def _get_vocab(self):
        fp = os.path.join(pathlib.Path(__file__).parent.parent, "freqs", "bnc_10k.json")
        freqs = load_json(fp)
        return list(freqs.keys())
