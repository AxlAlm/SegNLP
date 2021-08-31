
#basics
import numpy as np
import pandas as pd
from functools import lru_cache

# pytorch
import torch
from torch.nn.utils.rnn import pad_sequence


# segnlp
from segnlp import utils


class Level:

    def ___init__(self, 
                df: pd.DataFrame, 
                batch_size : int, 
                tasks : list,
                embs: np.ndarray =  None,
                device = None
                ):
        self._df = df
        self._device = device
        self._embs = embs
        self.batch_size = batch_size
        self.tasks = set(tasks)


    #@lru_cache(maxsize=None)
    def any_key(self, key:str):
        flat_values = self._df.loc[:, key].to_numpy()

        if isinstance(flat_values[0], str):
            return flat_values
        else:
            return torch.LongTensor(flat_values, device = self.device)


    def embs(self):

        if self._embs is None:
            raise KeyError

        return torch.FloatTensor(self._embs, device = self.device)

    @lru_cache(maxsize=None)
    def __getitem__(self, key:str):

        if getattr(self, key):
            return getattr(self, key)()
        else:
            return torch.pad_sequence(
                                torch.split(self.any_key(key), self.lengths()), 
                                batch_first = True,
                                pad_value = -1 if key in self.tasks else 0
                                )

  
class TokenLevel(Level):

    def ___init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    #@lru_cache(maxsize=None)
    def lengths(self):
        data = self._df.groupby(level=0, sort = False).size()
        return torch.LongTensor(data, device = self.device)

    #@lru_cache(maxsize=None)
    def mask(self):
        data = utils.create_mask(self.lengths(), as_bool = True) 
        return torch.BoolTensor(data, device = self.device)



class SegLevel(Level):
    
    def ___init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = "seg_id"


    #@lru_cache(maxsize=None)
    def lengths(self):
        data = self._df.groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()
        return torch.LongTensor(data, device = self.device)


    #@lru_cache(maxsize=None)
    def lengths_tok(self):
        seg_tok_lengths_flat = torch.LongTensor(self._df.groupby("seg_id", sort=False).to_numpy(), device = self.device)
        return torch.pad_sequence(
                            torch.split(seg_tok_lengths_flat, self.lengths()), 
                            batch_first = True,
                            pad_value = 0
                            )
        
    #@lru_cache(maxsize=None)
    def span_idxs(self):

        start_tok_ids = self._df.groupby(self.key, sort=False).first()["token_id"].to_numpy()
        end_tok_ids = self._df.groupby(self.key, sort=False).last()["token_id"].to_numpy()

        span_idxs_flat = np.concatenate(start_tok_ids,end_tok_ids, axis = 1)
        
        span_idxs = np.zeros((len(self.batch_size), np.max(self.lengths()), 2))
        
        floor = 0
        for i,l in enumerate(self.lengths()):
            span_idxs[i][:l] = span_idxs_flat[floor:floor+l]
            floor += l

        return torch.LongTensor(span_idxs, device = self.device)



class SpanLevel(SegLevel):

   def ___init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = "span_id"


class AMLevel(SegLevel):

   def ___init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = "am_id"



# class TSegLevel(Level):
    
#     def ___init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.key = "T-seg_id"


class BatchInput(dict):


    def __init__(self, 
                df: pd.DataFrame, 
                batch_size: int,
                word_embs: np.ndarray = None,
                seg_embs: np.ndarray = None,
                device = None
                ):
        
        self._df = df
        self.device = device
 
        self["token"] = TokenLevel(
                                    self._df, 
                                    batch_size = batch_size,
                                    word_embs = word_embs,
                                    device = device,
                                    )
        self["seg"] = SegLevel(
                                self._df, 
                                batch_size = batch_size,
                                seg_embs = seg_embs,
                                device = device,
                                )

        self["span"] = SpanLevel(
                                self._df,
                                batch_size = batch_size,
                                device = device,
                                )

        self["am"] = SpanLevel(
                                self._df,
                                batch_size = batch_size,
                                device = device,
                                )


    def to(self, device):
        self.device = device

