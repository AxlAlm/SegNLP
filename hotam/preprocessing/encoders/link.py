
#basics
from typing import List, Union, Dict
import numpy as np
from collections import Counter

#hotam
from hotam.preprocessing.encoders.base import Encoder


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


    def encode_list(self, item_list:List[str]) -> List[int]:
        return np.array([i + int(item) for i,item in enumerate(item_list)])
        

    def decode_list(self, item_list:List[str], pad=False) -> List[List[int]]:
        return np.array([str(int(item)-i) for i,item in enumerate(item_list)])


    def decode_token_links(self, item:List[str], span_token_lengths:List[int], none_spans:List[int]) -> List[int]:

        # first we collect the spans labels form the token labels
        # NOTE! decodding and encoding of links can only be done between labeled spans, e.g. spans that have labels
        # we need to then filter out spans with no labels
        start = 0
        j = 0
        idx_mapping = []
        span_items = []
        for i,length in enumerate(span_token_lengths):

            if none_spans[i]:
                span = item[start:start+length]
                majority_label = Counter(span).most_common(1)[0][0]
                span_items.append(majority_label)
                idx_mapping.append(j)
                j += 1
            else:
                idx_mapping.append(None)

            start += length 
                
        decoded_links = self.decode_list(span_items)
        #print(item, span_items, decoded_links, none_spans)

        #then we reconstruct the token labels from the decoded span labels
        decoded = []
        for i,j in enumerate(idx_mapping):
            if j is None:
                decoded.extend([0]*span_token_lengths[i])
            else:
                decoded.extend([decoded_links[j]]*span_token_lengths[i])


        return decoded
