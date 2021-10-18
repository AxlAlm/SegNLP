

#basics
import pandas as pd
from functools import wraps
from typing import Sequence, Union, List, Dict


# pytorch
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


# segnlp
from segnlp.data import Sample


class Batch:


    def __init__(self, 
                samples : List[Sample],
                device = None
                ):

        #
        self._samples = self._samples

        # device
        self.device = device

        # levels which are ok
        self.__ok_levels : set = set(["seg", "token", "span", "pair", "am", "adu"])

        # if we should use target segments for this batch
        self.use_target_segs : bool = False

        # init cache
        self.__cache : Dict[tuple, Union[Sequence, Tensor]] = {}

    
 
    def __len__(self):
        return len(self.samples)
 
    @property
    def samples(self):
        return self._samples

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
            flat : bool = False, 
            pred : bool = False,
            bidir : bool = True
            ):

        # create a key for string in cache
        cache_key = (level, key, flat, pred, bidir)

        # fetched cached data
        if cache_key in self.__cache:
            return self.__cache[cache_key]


        if level not in self.__ok_levels:
            raise KeyError

        data = [sample.get(level, key, pred = pred, bidir = bidir) for sample in self._samples]

        try:
            data = torch.tensor(data)
            pad_sequence(data, batch_first = True, padding_value = 0)
            data.to(self.device)
        except:
            pass

        
        self.__cache[cache_key] = data

        return data


    def add(self, level:str, key:str, data: Union[Sequence, Tensor]):


        if key not in self.label_encoder.task_labels:
            raise KeyError(f"cannot add values to key ='{key}'")

        (sample.add(level, key, data) for sample in self._sample)



    def to(self, device):
        self.device = device
