
from typing import List, Union, Dict
import numpy as np


class Encoder:

    def __len__(self):
        raise NotImplementedError()
    
    @property
    def keys(self):
        raise NotImplementedError()

    @property
    def name(self):
        return self._name

    def encode(self,item):
        raise NotImplementedError()

    def decode(self,item):
        raise NotImplementedError()

    def encode_list(self, item_list:List[str]) -> List[int]:
        """encodes list and if pad==true peforms padding.

        Parameters
        ----------
        item_list : List[str]
            list of strings to encode
        pad : bool, optional
            if one want to pad, by default False

        Returns
        -------
        List[int]
            list of encoded items (if set, with padding)
        """
        # if pad:
        #     padded = np.zeros((self.max_sample_length,))
        #     padded.fill(self.pad_value)
        #     for i, item in enumerate(item_list):
        #         padded[i] = self.encode(item)
        #     return padded
        # else:
        return np.array([self.encode(item) for item in item_list])


    def decode_list(self, item_list:List[int]) -> List[int]:
        """decodes a list of int int strings

        Parameters
        ----------
        item_list : List[int]
            items to decode
        pad : bool, optional
            if one wants to pad or not, by default False

        Returns
        -------
        List[int]
            list of decoded strings
        """
        return np.array([self.decode(item) for item in item_list])
