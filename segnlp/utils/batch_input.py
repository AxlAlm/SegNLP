
#basics
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
from cached_property import cached_property
import re

# pytorch
import torch
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


    def any_key(self, key:str):


        if isinstance(self, TokenLevel):
            flat_values = self._df.loc[:, key].to_numpy()
        else:
            flat_values = self._df.groupby(self.key, sort = False).first().loc[:, key].to_numpy()

        if isinstance(flat_values[0], str):
            return flat_values
        else:
            return torch.LongTensor(flat_values, device = self.device)


    def mask(self):
        return utils.create_mask(self.lengths(), as_bool = True).to(self.device)


    @utils.Memorize
    def __getitem__(self, key:str):

        if hasattr(self, key):
            return getattr(self, key)()

        elif "emb" in key:
            embs = torch.FloatTensor(self._pretrained_features[key], device = self.device)
            return embs[:, :self.max_len, :]
        
        elif "str" == key:
            return self.any_key(key)

        else:
            return pad_sequence(
                                torch.split(
                                            self.any_key(key), 
                                            utils.ensure_list(self.lengths())
                                            ), 
                                batch_first = True,
                                padding_value = -1 if self.task_regexp.search(key) else 0
                                )
        

class TokenLevel(Level):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        


    def lengths(self):
        data = self._df.groupby(level=0, sort = False).size().to_numpy()
        return torch.LongTensor(data, device = self.device)


class NonTokenLevel(Level):


    def lengths(self):
        data = self._df.groupby(level=0, sort=False)[self.key].nunique().to_numpy()
        return torch.LongTensor(data, device = self.device)


    def lengths_tok(self):

        seg_tok_lens = self._df.groupby(self.key, sort=False).size().to_numpy()
        seg_tok_lens = torch.LongTensor(seg_tok_lens, device = self.device)
        return pad_sequence(
                            torch.split(
                                        seg_tok_lens, 
                                        utils.ensure_list(self.lengths())
                                        ), 
                            batch_first = True,
                            padding_value = 0
                            )
        

    def span_idxs(self):

        start_tok_ids = self._df.groupby(self.key, sort=False).first()["sample_token_id"].to_numpy()
        end_tok_ids = self._df.groupby(self.key, sort=False).last()["sample_token_id"].to_numpy()

        span_idxs_flat = torch.LongTensor(np.column_stack((start_tok_ids, end_tok_ids)), device = self.device)

        sample_span_idxs = torch.split(span_idxs_flat, utils.ensure_list(self.lengths()))

        return pad_sequence(sample_span_idxs, batch_first=True)
    

class SegLevel(NonTokenLevel):
    
    def __init__(self, *args, **kwargs):
        self.key = "seg_id"
        super().__init__(*args, **kwargs)


class SpanLevel(NonTokenLevel):

    def __init__(self, *args, **kwargs):
        self.key = "span_id"
        super().__init__(*args, **kwargs)


class AMLevel(NonTokenLevel):

    def __init__(self, *args, **kwargs):
        self.key = "am_id"
        super().__init__(*args, **kwargs)


class ADULevel(NonTokenLevel):

    def __init__(self, *args, **kwargs):
        self.key = "adu_id"
        super().__init__(*args, **kwargs)


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

        self["am"] = AMLevel(
                                self._df,
                                batch_size = batch_size,
                                device = device,
                                )

        self["adu"] = ADULevel(
                                self._df,
                                batch_size = batch_size,
                                device = device,
                                )


    def to(self, device):
        self.device = device

