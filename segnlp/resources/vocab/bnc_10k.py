
#basics
import pathlib
import os


# segnlp
from segnlp.utils import load_json
from segnlp.resources.vocab import Vocab


def bnc_10k():

    fp = os.path.join(pathlib.Path(__file__).parent.parent, "freqs", "bnc_10k.json")
    freqs = load_json(fp)

    return Vocab(
                name = "bnc_10k",
                freq_dist = freqs,
                size = 10000,
                )