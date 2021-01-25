

from hotam.preprocessing.encoders.base import Encoder
from typing import List, Union, Dict
import numpy as np


class RelationEncoder(Encoder):


    """
        relation label encoding words a bit different as relations are assumed to be ints 
        telling us how many ACs back or forward the related AC is at. e.g. -1 means that the AC at index i is related to i-1.
        but when trying to predict these relations in a NN its usualy easier to treat relations as pointer. e.g. a relation == 3, means
        an AC is related to the ac at index == 3. So we convert relations from -1 to  index -1, so we can use e.g. Attention etc to predict ACs ( see Joint Pointer Network for example)
        
        for relation ids are hence indexes of max acs in the sample. I.e. all ACs in a sample are possible relations.

    """

    def __init__(self, name:str, max_back_relation:int, max_forward_relation:int, max_acs:int):
        self._name = name
        #max_relations = max_forward_relation - max_back_relation
        self._labels =[i for i in range(max_back_relation, max_forward_relation)]
        self._ids = [i for i in range(max_acs)]

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids


    def encode(self,item):
        raise TypeError("relation encoding cannot be done on a sperate label but need the whole sample")


    def decode(self,item):
        raise TypeError("relation decoding cannot be done on a sperate label but need the whole sample")


    def encode_list(self, item_list:List[str]) -> List[int]:
        return np.array([i + int(item) for i,item in enumerate(item_list)])
        

    def decode_list(self, item_list:List[str], pad=False) -> List[List[int]]:
        return np.array([str(int(item)-i) for i,item in enumerate(item_list)])
