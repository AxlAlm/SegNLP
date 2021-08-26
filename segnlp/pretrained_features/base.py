
# basic
import numpy as np
from abc import ABC, abstractmethod
import os
from typing import List, Dict, Tuple
import re
import copy
from pathlib import Path


# axl nlp
import segnlp
import segnlp.utils as u
from segnlp import get_logger



logger = get_logger("FEATURES")


class FeatureModel(ABC):

        
    @property
    def name(self):
        return self._name


    @property
    def feature_dim(self):
        return self._feature_dim


    @property
    def level(self):
        return self._level


    @property
    def dtype(self):
        return self._dtype


    @property
    def group(self):
        return self._group
    

    @property
    def context(self):
        if hasattr(self, "_context"):
           return self._context
        else:
            return False

    @abstractmethod
    def extract(self):
        pass

    @property
    def params(self):
        return {}