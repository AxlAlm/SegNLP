
from typing import List, Union, Dict
import numpy as np


from hotam.preprocessing.encoders.base import Encoder
from transformers import BertTokenizer


class BertTokEncoder(Encoder):
    """ Bert Tokenize Encoder"""

    def __init__(self):
        self._name = "bert_encs"
        #self.pad_value = 0
        self.tokenizer = BertTokenizer.from_pretrained(
                                                    'bert-base-uncased',
                                                    )
    
    def __len__(self):
        return len(self.tokenizer.get_vocab())
    

    @property
    def keys(self):
        return self.tokenizer.get_vocab()

    
    def decode(self, item):
        return self.tokenizer.decode([item])

    
    def decode_list(self, item_list):
        return self.tokenizer.decode(item)

    
    def encode(self, item:str) -> List[int]:
        """encodes word with bert tokenize.encode()

        Parameters
        ----------
        item : str
            word to encode

        Returns
        -------
        list
            list of encoding ids
        """
        enc_ids = self.tokenizer.encode(item, add_special_tokens=False)
        return enc_ids


    def encode_list(self, item_list:List[str]) -> List[List[int]]:
        """encodes a list of words with bert tokenize.encode()

        Parameters
        ----------
        item_list : List[str]
            words to encode
        pad : bool, optional
            if pad or not, by default True

        Returns
        -------
        List[List[int]]
            list of list of encoded words
        """
        item_string = " ".join(item_list)
        enc_ids = np.array(self.tokenizer.encode(   
                                                    item_string, 
                                                    add_special_tokens=False, 
                                                    #max_length=self.max_sample_length, 
                                                    #pad_to_max_length=True
                                                    ))
        return enc_ids
