

#basic
import numpy as np
import os
from typing import List, Union
import _pickle as pkl

#sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


#hotam
from hotam.features.base import FeatureModel
import hotam.utils as u
from hotam import get_logger

class DummyFeature(FeatureModel):

    def __init__(self, level="word"):

        #self.vocab = vocab
        self._name = f"dummy_{level}_feature"
        self._level = level
        self._feature_dim = 100
        self._dtype = np.float32
        self._group = f"{level}_embs"
    
    #@feature_memory
    def extract(self, df):
        
        if self._level == "word":
            return np.random.random((df.shape[0],100))
        else:
            return np.random.random(100)


