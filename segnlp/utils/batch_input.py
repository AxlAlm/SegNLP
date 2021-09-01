
#basics
import numpy as np
import pandas as pd
from cached_property import cached_property
import re

# pytorch
import torch
from torch.nn.functional import batch_norm
from torch.nn.utils.rnn import pad_sequence


# segnlp
from segnlp import utils


class Level:

    def __init__(self, 
                df: pd.DataFrame, 
                batch_size : int, 
                pretrained_features:dict = {},
                device = None
                ):
        self._df = df
        self._pretrained_features = pretrained_features
        self.device = device
        self.batch_size = batch_size
        self.task_regexp = re.compile("seg|link|label|link_label")
        self.max_len = torch.max(self.lengths())

    #@lru_cache(maxsize=None)
    def any_key(self, key:str):
        flat_values = self._df.loc[:, key].to_numpy()

        if isinstance(flat_values[0], str):
            return flat_values
        else:
            return torch.LongTensor(flat_values, device = self.device)


    @utils.Memorize
    def __getitem__(self, key:str):

        print("HELLLOLOLOLOLOLO")

        if hasattr(self, key):
            return getattr(self, key)()

        elif "emb" in key:
            embs = torch.FloatTensor(self._pretrained_features[key], device = self.device)
            return embs[:, :self.max_len, :]
        
        elif "token" == key:
            return self.any_key(key)

        else:
            return pad_sequence(
                                torch.split(self.any_key(key), self.lengths()), 
                                batch_first = True,
                                pad_value = -1 if self.task_regexp.search(key) else 0
                                )

  
class TokenLevel(Level):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    #@lru_cache(maxsize=None)
    def lengths(self):
        data = self._df.groupby(level=0, sort = False).size().to_numpy()
        return torch.LongTensor(data, device = self.device)

    #@lru_cache(maxsize=None)
    def mask(self):
        data = utils.create_mask(self.lengths(), as_bool = True) 
        return torch.BoolTensor(data, device = self.device)



class SegLevel(Level):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = "seg_id"


    #@lru_cache(maxsize=None)
    def lengths(self):
        data = self._df.groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()
        return torch.LongTensor(data, device = self.device)


    #@lru_cache(maxsize=None)
    def lengths_tok(self):
        seg_tok_lengths_flat = torch.LongTensor(self._df.groupby("seg_id", sort=False).to_numpy(), device = self.device)
        return pad_sequence(
                            torch.split(seg_tok_lengths_flat, self.lengths()), 
                            batch_first = True,
                            pad_value = 0
                            )
        
    #@lru_cache(maxsize=None)
    def span_idxs(self):

        start_tok_ids = self._df.groupby(self.key, sort=False).first()["sample_token_id"].to_numpy()
        end_tok_ids = self._df.groupby(self.key, sort=False).last()["sample_token_id"].to_numpy()

        span_idxs_flat = torch.LongTensor(np.column_stack((start_tok_ids, end_tok_ids)) ,device = self.device)

        sample_span_idxs = torch.split(span_idxs_flat, utils.ensure_numpy(self.lengths()).tolist())

        return pad_sequence(sample_span_idxs, batch_first=True)
    


class SpanLevel(SegLevel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = "span_id"


class AMLevel(SegLevel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = "am_id"



class BatchInput(dict):


    def __init__(self, 
                df: pd.DataFrame, 
                batch_size: int,
                pretrained_features: dict = {},
                device = None
                ):
        
        self._df = df
        self.device = device
 
        self["token"] = TokenLevel(
                                    self._df, 
                                    batch_size = batch_size,
                                    pretrained_features = pretrained_features,
                                    device = device,
                                    )
        self["seg"] = SegLevel(
                                self._df, 
                                batch_size = batch_size,
                                pretrained_features = pretrained_features,
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

