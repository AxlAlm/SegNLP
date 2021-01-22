

#basic
import numpy as np
import os
from typing import List, Union
import _pickle as pkl

#torch
import torch
from torch.nn.functional import one_hot

#hotam
from hotam.features.base import FeatureModel, feature_memory
import hotam.utils as u
from hotam import get_logger
from hotam.resources.deps import dep_labels_eng
from hotam.resources.pos import pos_labels
from hotam.preprocessing import DataSet


class OneHots(FeatureModel):

    def __init__(self, label:str):

        #self.vocab = vocab
        self._name = label
        self._level = "word"

        if label == "pos":
            labels = pos_labels
        elif label == "deprel":
            labels  = dep_labels_eng
        else:
            raise NotImplementedError(f"""  {label} is not a supported feature. Note that the thing you encode to one_hot needs to exist in the dataset columns. 
                                            Also, words are not allowed to be one_hot encoded as the feature_dim will then be == vocab which is over 300k. Vocab is predefined.
                                            use BOW instead if you want bag of word feature""")

        self.label = label
        self._feature_dim = len(labels)
        self._dtype = np.uint8
    
    #@feature_memory
    def extract(self, df):
        ids = df[self.label].to_numpy()
        ids_t = torch.LongTensor(ids)
        one_hots = torch.nn.functional.one_hot(ids_t, num_classes=self._feature_dim).numpy().astype(self._dtype)
        return one_hots
