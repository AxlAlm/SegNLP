

import numpy as np
from collections import defaultdict 


#pytroch 
import torch

#segnlp 
from segnlp.utils import dynamic_update, tensor_dtype
from segnlp.visuals.tree_graph import arrays_to_tree
from segnlp.utils import ensure_numpy

class Input(dict):


    def __init__(self,):
        super().__init__()
        self.current_epoch = None
        self.label_pad_value = -1
        self._size = 0
        self._ids = []
        self.oo = []


    def __len__(self):
        return len(self._ids)


    @property
    def ids(self):
        return self._ids

    @property
    def levels(self):
        return list(self.keys())


    def to(self, device):
        self.device = device
        for level in self:
            for k,v in self[level].items():
                if torch.is_tensor(v):
                    self[level][k] = v.to(self.device)
                    
        return self
    

    def to_tensor(self, device="cpu"):

        for level in self:
            for k,v in self[level].items():

                if k == "text":
                    continue

                self[level][k] = torch.tensor(v, dtype=tensor_dtype(v.dtype), device=device)
        return self


    def to_numpy(self):

        self._ids = np.array(self._ids)
        for level in self:

            for k,v in self[level].items():
                
                if isinstance(v, np.ndarray):
                    continue
                
                if isinstance(v[0], np.ndarray):
                    dtype = v[0].dtype
                else:
                    dtype = np.int

                self[level][k] = np.array(v, dtype=dtype)

        return self


    def change_pad_value(self, level:str, task:str, new_value:int):
        self[level][task][self[level][task] == self.label_pad_value] = new_value
     

    def add(self, k, v, level, pad_value=0):
        # if k not in self:
        #     self[k] = [v]
        # else:
        #     self[k].append(v)
        # if k == "id":
        #     if "id" in k:
        #         self[k] = [v]
        #     else:
        #         self[k].append(v)


        if k == "ids":
            self._ids.append(v)
            return
        
        if level not in self:
            self[level] = {}
         
        if k not in self[level]:
            if isinstance(v, np.ndarray):
                self[level][k] = np.expand_dims(v, axis=0)
            else:
                self[level][k] = [v]
        else:
            if isinstance(v, int):
                self[level][k].append(v)
            else:
                self[level][k] = dynamic_update(self[level][k], v, pad_value=pad_value)


    def sort(self):
        lengths = self["token"]["lengths"]
        
        lengths_decending = np.argsort(lengths)[::-1]

        #r = np.arange(lengths_decending.shape[0])
        #si = np.argsort(a)
        self.oo = np.argsort(lengths_decending)

        for group in self:
            for k, v in self[group].items():
                self[group][k] = v[lengths_decending]
