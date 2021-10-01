# basics
import os
import numpy as np
import pandas as pd


#allennlp
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids


# segnlp
from segnlp import utils


class ELMoEmbeddings():

    """
    https://allennlp.org/elmo

    """
    
    def __init__(self):
    
        #options_file_http = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        #small_weight_http = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        
        
        options_http_original = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
        weight_http_original = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
        
        
        options_fp = "/tmp/elmo_options_original.json"
        weights_fp = "/tmp/elmo_weights_original.hdf5"

        utils.download(options_http_original, options_fp)
        utils.download(weight_http_original, weights_fp)

        self.elmo = Elmo(
                options_file = options_fp,
                weight_file = weights_fp,
                num_output_representations = 1
                )

        self._name = "elmo"
        self._context = "sentence"
        self._level = "word"
        self._dtype = np.float32
        self._group = "word_embs"
        self._feature_dim = 1024


    @utils.timer
    def extract(self, df) -> np.ndarray:
        tokens = df["str"].to_numpy()
        batch_char_ids = batch_to_ids([tokens])
        embs = self.elmo(batch_char_ids)["elmo_representations"][0][0]
        return utils.ensure_numpy(embs)