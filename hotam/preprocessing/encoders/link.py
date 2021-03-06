
#basics
from typing import List, Union, Dict
import numpy as np
from collections import Counter

#hotam
from hotam.preprocessing.encoders.base import Encoder
from hotam.utils import ensure_numpy

class LinkEncoder(Encoder):


    """
        relation label encoding words a bit different as relations are assumed to be ints 
        telling us how many ACs back or forward the related AC is at. e.g. -1 means that the AC at index i is related to i-1.
        but when trying to predict these relations in a NN its usualy easier to treat relations as pointer. e.g. a relation == 3, means
        an AC is related to the ac at index == 3. So we convert relations from -1 to  index -1, so we can use e.g. Attention etc to predict ACs ( see Joint Pointer Network for example)
        
        for relation ids are hence indexes of max acs in the sample. I.e. all ACs in a sample are possible relations.

    """

    def __init__(self, name:str, max_spans:int):
        self._name = name
        self._labels = [i for i in range(-int(max_spans/2), int(max_spans/2))]
        self._ids = [i for i in range(max_spans)]

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


    def encode_list(self, item_list:np.ndarray) -> np.ndarray:
        a = ensure_numpy(item_list)
        idx = np.arange(a.shape[0])
        return idx + a
        #return np.array([i + int(item) for i,item in enumerate(item_list)])
        

    def decode_list(self, item_list:np.ndarray, pad=False) -> np.ndarray:
        a = ensure_numpy(item_list)
        idx = np.arange(a.shape[0])
        return a - idx
        #return np.array([str(int(item)-i) for i,item in enumerate(item_list)])


    # def decode_token_links(self, item:List[str], span_token_lengths:List[int], none_spans:List[int]) -> List[int]:

    #     # first we collect the spans labels form the token labels
    #     # NOTE! decodding and encoding of links can only be done between labeled spans, e.g. spans that have labels
    #     # we need to then filter out spans with no labels
    #     start = 0
    #     j = 0
    #     idx_mapping = []
    #     span_items = []
    #     for i,length in enumerate(span_token_lengths):

    #         if none_spans[i]:
    #             span = item[start:start+length]
    #             majority_label = Counter(span).most_common(1)[0][0]
    #             span_items.append(majority_label)
    #             idx_mapping.append(j)
    #             j += 1
    #         else:
    #             idx_mapping.append(None)

    #         start += length 
                
    #     decoded_links = self.decode_list(span_items)
    #     #print(item, span_items, decoded_links, none_spans)

    #     #then we reconstruct the token labels from the decoded span labels
    #     decoded = []
    #     for i,j in enumerate(idx_mapping):
    #         if j is None:
    #             decoded.extend(["0"]*span_token_lengths[i])
    #         else:
    #             decoded.extend([decoded_links[j]]*span_token_lengths[i])


    #     return decoded

    def __create_token_idx_map(self, x, span_token_lengths, none_spans):
        """
        As decoding and encoding links are done by subtracting or ... the index of units with the link labels
        we need to create a index over units that stretch over a array of tokens. E.g. if tokens between i;j are of unit idx x
        we need to create a array with shape equal to the token array where i:j = x.
        """
        x = ensure_numpy(x)
        span_token_lengths = ensure_numpy(span_token_lengths)
        none_spans = ensure_numpy(none_spans)
        token_unit_idx = np.zeros(x.shape)
        start = 0
        nr_units = 0
        for i in range(span_token_lengths.shape[0]):
            if span_token_lengths[i]:
                token_unit_idx[start:start+span_token_lengths[i]] = nr_units

        return token_unit_idx

    def encode_token_links(self, x:np.ndarray, span_token_lengths:np.ndarray, none_spans:np.ndarray) -> List[int]:
        idx = self.__create_token_idx_map(x, span_token_lengths, none_spans)
        return idx + x


    def decode_token_links(self, x:np.ndarray, span_token_lengths:np.ndarray, none_spans:np.ndarray) -> List[int]:
        idx = self.__create_token_idx_map(x, span_token_lengths, none_spans)
        return x - idx


