
#basics
import numpy as np
import pandas as pd
import re

# pytorch
import torch
from torch import Tensor
from torch._C import dtype
from torch.nn.utils.rnn import pad_sequence

# segnlp
from segnlp import utils

class BatchInput:


    def __init__(self, 
                df: pd.DataFrame, 
                pretrained_features: dict = {},
                device = None
                ):
        self._df = df
        self._task_regexp = re.compile("seg|link|label|link_label")
        self._pretrained_features = pretrained_features
        self.device = device
        self.__ok_levels = set(["seg", "token", "span"])

        if "am_id" in self._df.columns:
            self.__ok_levels.update(["am_id", "adu_id"])

        self._size = self._df["sample_id"].nunique()


    def __len__(self):
        return self._size
 

    def _get_column_values(self, level: str, key:str):

        if level == "token":
            flat_values = self._df.loc[:, key].to_numpy()
        else:
            flat_values = self._df.groupby(f"{level}_id", sort = False).first().loc[:, key].to_numpy()

        if isinstance(flat_values[0], str):
            return flat_values
        else:
            return torch.LongTensor(flat_values)


    def _get_span_idxs(self, level):

        if level == "am":
            ADU_start = self._df.groupby("adu_id", sort=False).first()["sample_token_id"].to_numpy()
            ADU_end = self._df.groupby("adu_id", sort=False).last()["sample_token_id"].to_numpy() + 1 

            AC_lens = self._df.groupby("seg_id", sort=False).size().to_numpy()

            AM_start = ADU_start
            AM_end = ADU_end - AC_lens

            return torch.LongTensor(np.column_stack((AM_start, AM_end)))

        else:

            start_tok_ids = self._df.groupby(f"{level}_id", sort=False).first()["sample_token_id"].to_numpy()
            end_tok_ids = self._df.groupby(f"{level}_id", sort=False).last()["sample_token_id"].to_numpy() + 1

            return torch.LongTensor(np.column_stack((start_tok_ids, end_tok_ids)))


    def _get_mask(self, level):
        return utils.create_mask(self.get(level, "lengths"), as_bool = True)


    def _get_lengths(self, level):

        if level == "token":
            return torch.LongTensor(self._df.groupby(level=0, sort = False).size().to_numpy())
        else:
            return torch.LongTensor(self._df.groupby(level=0, sort=False)[f"{level}_id"].nunique().to_numpy())


    def _get_pretrained_embeddings(self, level:str, flat:bool):

        if level == "token":
            embs = self._pretrained_features["word_embs"]
        else:
            embs = self._pretrained_features["seg_embs"]

        embs = embs[:, :max(self._get_lengths(level)), :]

        if flat:
            embs = embs[self._get_mask("level")]

        return torch.tensor(embs, dtype = torch.float)


    @utils.Memorize
    def get(self, 
            level:str, 
            key:str, 
            flat:bool = False, 
            ):

        if level not in self.__ok_levels:
            raise KeyError


        if key == "lengths":
            data =  self._get_lengths(level)

        elif key == "embs":
            data =  self._get_pretrained_embeddings(level, flat=flat)

        elif key == "mask":
            data = self._get_mask(level)

        else:
            if key == "span_idxs":
                data = self._get_span_idxs(level)
            else:
                data = self._get_column_values(level, key)


            if isinstance(data[0], str):
                return data

            if not flat:

                if level == "am" and key == "span_idxs":
                    level = "adu"
            
                lengths = utils.ensure_list(self.get(level, "lengths"))
                    
                data =  pad_sequence(
                                    torch.split(
                                                data, 
                                                lengths
                                                ), 
                                    batch_first = True,
                                    padding_value = -1 if self._task_regexp.search(key) else 0,
                                    )
        
        if isinstance(data, Tensor):
            data = data.to(self.device)

        return data


    def to(self, device):
        self.device = device

        # for k, level_class in self.items():
        #     level_class.device = device

