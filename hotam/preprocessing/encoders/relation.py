

from hotam.preprocessing.encoders.base import Encoder
from typing import List, Union, Dict
import numpy as np


class RelationEncoder(Encoder):

    def __init__(self, name:str):
        self._name = name

    def encode(self,item):
        raise TypeError("relation encoding cannot be done on a sperate label but need the whole sample")


    def decode(self,item):
        raise TypeError("relation decoding cannot be done on a sperate label but need the whole sample")


    def encode_list(self, item_list:List[str]) -> List[int]:
        return np.array([i + int(item) for i,item in enumerate(item_list)])
        

    def decode_list(self, item_list:List[str], pad=False) -> List[List[int]]:
        return np.array([str(int(item)-i) for i,item in enumerate(item_list)])
