
#basics
import numpy as np
import pandas as pd
from functools import wraps


# pytorch
import torch

# segnlp
from segnlp import utils


class Level:

    def ___init__(self, 
                df: pd.DataFrame, 
                batch_size : int, 
                embs: np.ndarray =  None,
                device = None
                ):
        self._df = df
        self._device = device
        self._cache = {}
        self._embs = embs
        self.batch_size = batch_size


    def cache(f):
        @wraps(f) #needed to accurately perserve the function name and doc of the function it decorates
        def tensor_cache(self, *args):

            if f.__name__ == "any_key":
                key = args[0]
            else:
                key = f.__name__

            if key not in self._cache:
                
                data  = f(self)
                
                if key != "token":
                    data = torch.LongTensor(data, device = self.device)
                
                self.cache[key] =data

            return self.cache[key]

        return tensor_cache


    @cache
    def any_key(self, key):
        flat_values = self._df.loc[:, key].to_numpy()
        splits = np.split(flat_values, self.lengths())
        return utils.pad(splits, pad_value = -1 if key in tasks else 0)


    def embs(self):

        if self._embs is None:
            raise KeyError

        return torch.FloatTensor(self._embs, device = self.device)

    
    def __getitem__(self,key):

        if getattr(self, key):
            return getattr(self, key)()
        else:
            return self.any_key(key)


class TokenLevel(Level):

    def ___init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    @cache
    def lengths(self):
        return self._df.groupby(level=0, sort = False).size()

    @cache
    def mask(self):
        return utils.create_mask(self.lengths(), as_bool = True) 


class SegLevel(Level):
    
    def ___init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = "seg_id"


    @cache
    def lengths(self):
        return self._df.groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()

    @cache
    def lengths_tok(self):
        seg_tok_lengths_flat = self._df.groupby("seg_id", sort=False).to_numpy()
        seg_tok_lengths = np.split(seg_tok_lengths_flat, self.lengths())
        return utils.pad(seg_tok_lengths, 0)

        
    @cache
    def span_idxs(self):

        start_tok_ids = self._df.groupby(self.key, sort=False).first()["token_id"].to_numpy()
        end_tok_ids = self._df.groupby(self.key, sort=False).last()["token_id"].to_numpy()

        span_idxs_flat = np.concatenate(start_tok_ids,end_tok_ids, axis = 1)
        
        span_idxs = np.zeros((len(self.batch_size), np.max(self.lengths()), 2))
        
        floor = 0
        for i,l in enumerate(self.lengths()):
            span_idxs[i][:l] = span_idxs_flat[floor:floor+l]
            floor += l

        return span_idxs



class SpanLevel(SegLevel):

   def ___init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = "span_id"


class AMLevel(SegLevel):

   def ___init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = "am_id"



class Batch(dict):


    def __init__(self, 
                df: pd.Dataframe, 
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

