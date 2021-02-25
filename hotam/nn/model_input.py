

import numpy as np

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


    def __len__(self):
        return self["id"].shape[0]


    def to(self, device):
        self.device = device
        for k,v in self.items():
            if torch.is_tensor(v):
                self[k] = v.to(self.device)
        return self
    

    def to_tensor(self):
        for k,v in self.items():
            self[k] = torch.tensor(v, dtype=tensor_dtype(v.dtype))
        return self


    def to_numpy(self):
        for k,v in self.items():

            if isinstance(v, np.ndarray):
                continue
            
            if isinstance(v[0], np.ndarray)
                dtype = v[0].dtype
            else:
                dtype = np.int

            self[k] = np.array(v, dtype=dtype)
        return self


    def change_pad_value(self, task, new_value):
        #for task in self.all_tasks:
        self[task][self[task] == self.pad_value] = new_value
            # self[task][~self[f"{self.prediction_level}_mask"].type(torch.bool)] = -1
     

    def add(self, k, v):
        # if k not in self:
        #     self[k] = [v]
        # else:
        #     self[k].append(v)
         
        if k not in self:
            if isinstance(v, np.ndarray):
                self[k] = np.expand_dims(v, axis=0)
            else:
                self[k] = [v]
        else:
            if isinstance(v, int):
                self[k].append(v)
            else:
                self[k] = dynamic_update(self[k],v)


