    
    
from hotam.preprocessing.encoders.label import LabelEncoder
import numpy as np
from typing import List, Union, Dict


class PeLinkEncoder(LabelEncoder):


    def encode_list(self, item_list:List[str]) -> List[List[int]]:
        
        # if pad:
        #     padded = np.zeros((self.max_sample_length,))
        #     padded.fill(self.pad_value)
        #     for i, item in enumerate(item_list):
        #         padded[i] = i + int(item)
        #     return padded
        # else:
        return np.array([i + int(item) for i,item in enumerate(item_list)])
        

    def decode_list(self, item_list:List[str], pad=False) -> List[List[int]]:
        return np.array([str(item-i) for i,item in enumerate(item_list) if item != self.pad_value])
