

import numpy as np
from collections import defaultdict 


#pytroch 
import torch

#hotam 
from hotam.utils import dynamic_update, tensor_dtype


class ModelInput(dict):


    def __init__(self, 
                #all_tasks:list
                ):
        super().__init__()
        #self.all_tasks = all_tasks
        self.current_epoch = None
        self.pad_value = 0
        self._size = 0
        self._ids = []


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
    

    def to_tensor(self):

        for level in self:
            for k,v in self[level].items():

                if k == "text":
                    continue
                self[level][k] = torch.tensor(v, dtype=tensor_dtype(v.dtype))
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
        #for task in self.all_tasks:
        self[level][task][self[level][task] == self.pad_value] = new_value
            # self[task][~self[f"{self.prediction_level}_mask"].type(torch.bool)] = -1
     

    def add(self, k, v, level):
        # if k not in self:
        #     self[k] = [v]
        # else:
        #     self[k].append(v)
        # if k == "id":
        #     if "id" in k:
        #         self[k] = [v]
        #     else:
        #         self[k].append(v)


        if k == "id":
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
                self[level][k] = dynamic_update(self[level][k], v)
