

#basics
from numpy.lib.arraysetops import isin
import pandas as pd
from functools import wraps
from typing import Sequence, Union, List, Dict


# pytorch
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


# segnlp
from segnlp.data import Sample
from segnlp import utils


class Batch:


    def __init__(self, 
                samples : List[Sample],
                device = None
                ):

        #target and pred samples
        self._target_samples = samples
        self._pred_samples = [s.copy(clean=True) for s in samples]

        # set task labels
        self.task_labels = samples[0]._task_labels

        # device
        self.device = device

        # if we should use target segments for this batch
        self.use_target_segs : bool = False

        # init cache
        self.__cache : Dict[tuple, Union[Sequence, Tensor]] = {}

    
    def __len__(self):
        return len(self._target_samples)


    def __use_targets_wrapper(func):

        """
        If batch is suppose to use target segments instead of predicted segemnts
        we swap the input value to get(). This is so we dont have to change in our
        model code.
        
        """
        
        @wraps(func)
        def wrapped_get(self, *args, **kwargs):
            
            if self.use_target_segs:
                kwargs["pred"] = False

            return func(self, *args, **kwargs)
        
        return wrapped_get


    @__use_targets_wrapper
    def get(self, 
            level : str, 
            key : str, 
            pred : bool = False,
            bidir : bool = True,
            flat : bool = False
            ):

        # create a key for string in cache
        cache_key = (level, key, pred, bidir, flat) 

        samples = self._pred_samples if pred else self._target_samples

        # fetched cached data
        if cache_key in self.__cache:
            return self.__cache[cache_key]

        # create a mask from lengths
        if key == "mask":
            data = utils.create_mask([sample.get(level, "length") for sample in samples])


        else:
            data = [sample.get(level, key) for sample in samples]

            if flat:
                data = [item for sublist in data for item in sublist]


            if not isinstance(data[0], int):
                
                # for creatin keys we get string
                try:
                    data = pad_sequence(data, batch_first = True, padding_value = -1 if key in self.task_labels else 0)
                except TypeError:
                    pass

            else:
                data = torch.LongTensor(data)


        # check to see if we can add data to tensor
        try:
            data.to(self.device)
        except AttributeError:
            pass
  
        self.__cache[cache_key] = data

        return data


    def add(self, level:str, key:str, data: Union[Sequence, Tensor]):


        if key not in self.task_labels:
            raise KeyError(f"cannot add values to key ='{key}'")

        for i,sample in enumerate(self._pred_samples):
            sample.add(level, key, data[i]) 


    def to(self, device):
        self.device = device
