
#basic
import numpy as np
import os
from typing import List

#alxnlp
from hotam.features.base import FeatureModel
import hotam.utils as u
from hotam.preprocessing.resources.vocab import vocab


class OneHot(FeatureModel):

    """
    Class for Bag Of Word features.

    """

    def __init__(self):
        self.vocab = vocab
        self._name = "onehot"
        self._feature_dim = len(self.vocab)
        self._level = "doc"
        self._dtype = np.uint8

   
    def extract(self, df):
        tokens = df["text"].to_numpy()

        onehot = np.zeros(len(self.vocab))

        for word in words:
            onehot[self.vocab.get(word,1)] = 1

        return onehot        