
#basics
from gensim import utils
import numpy as np
from numpy.lib.utils import deprecate
import pandas as pd
import re
from functools import wraps
from typing import Union


# pytorch
import torch
from torch import Tensor
from torch._C import dtype
from torch.nn.utils.rnn import pad_sequence

# segnlp
from .label_encoder import LabelEncoder
from .array import ensure_numpy
from .array import ensure_list
from .array import create_mask
from .array import np_cumsum_zero
from .find_overlap import find_overlap
from .misc import timer


class Batch:


    def __init__(self, 
                df: pd.DataFrame, 
                label_encoder : LabelEncoder, 
                pretrained_features: dict = {},
                device = None
                ):

        self._df : pd.DataFrame = df
        self._pred_df : pd.DataFrame = df.copy(deep=True)

        if "seg" in label_encoder.task_labels:
            self._pred_df["seg_id"] = None

        self._pred_df["target_id"] = None
        for task in label_encoder.task_labels:
            self._pred_df[task] = None

        #
        self.label_encoder : LabelEncoder = label_encoder
        self._task_regexp = re.compile("seg|link|label|link_label")
        self._pretrained_features = pretrained_features
        self.device = device
        self.__ok_levels = set(["seg", "token", "span", "pair"])

        if "am_id" in self._df.columns:
            self.__ok_levels.update(["am_id", "adu_id"])

        self._size = self._df["sample_id"].nunique()

        self.use_target_segs : bool = False

        # cache
        self.__cache = {}


    def __len__(self):
        return self._size
 

    def __sampling_wrapper(func):
        
        @wraps(func)
        def wrapped_get(self, *args, **kwargs):
            
            if self.use_target_segs:
                kwargs["pred"] = False

            return func(self, *args, **kwargs)
        
        return wrapped_get

    
    def __get_column_values(self, df: pd.DataFrame, level: str, key:str):

        if level == "token":
            flat_values = df.loc[:, key].to_numpy()
        else:
            flat_values = df.groupby(f"{level}_id", sort = False).first().loc[:, key].to_numpy()

        if isinstance(flat_values[0], str):
            return flat_values
        else:
            return torch.LongTensor(flat_values)


    def __get_span_idxs(self, df: pd.DataFrame, level:str ):

        if level == "am":
            ADU_start = df.groupby("adu_id", sort=False).first()["sample_token_id"].to_numpy()
            ADU_end = df.groupby("adu_id", sort=False).last()["sample_token_id"].to_numpy() + 1 

            AC_lens = df.groupby("seg_id", sort=False).size().to_numpy()

            AM_start = ADU_start
            AM_end = ADU_end - AC_lens

            return torch.LongTensor(np.column_stack((AM_start, AM_end)))

        else:

            start_tok_ids = df.groupby(f"{level}_id", sort=False).first()["sample_token_id"].to_numpy()
            end_tok_ids = df.groupby(f"{level}_id", sort=False).last()["sample_token_id"].to_numpy() + 1

            return torch.LongTensor(np.column_stack((start_tok_ids, end_tok_ids)))


    def __get_mask(self, level:str, pred : bool = False):
        return create_mask(self.get(level, "lengths", pred = pred), as_bool = True)


    # def __seg_tok_lengths(self, df: pd.DataFrame, level:str):
    #     return df.groupby(level, sort=False).size().to_numpy()


    def __get_lengths(self, df: pd.DataFrame, level:str):

        if level == "token":
            return torch.LongTensor(df.groupby(level=0, sort = False).size().to_numpy())
        else:
            return torch.LongTensor(df.groupby(level=0, sort=False)[f"{level}_id"].nunique().to_numpy())


    def __get_pretrained_embeddings(self, df:pd.DataFrame, level:str, flat:bool):

        if level == "token":
            embs = self._pretrained_features["word_embs"]
        else:
            embs = self._pretrained_features["seg_embs"]

        embs = embs[:, :max(self.__get_lengths(df, level)), :]

        if flat:
            embs = embs[self.__get_mask("level")]

        return torch.tensor(embs, dtype = torch.float)


    def __add_link_matching_info(self, pair_df):


        def check_true_pair(row, mapping):

            p1 = row["p1"]
            p2 = row["p2"]
            dir = row["direction"]
            
            source = p2 if dir == 2 else p1
            target = p1 if dir == 2 else p2

            if source not in mapping:
                return False
            else:
                correct_target = mapping[source]
                return correct_target == target

        
        j_jt = self._df.loc[:, ["seg_id", "target_id"]].dropna()

        # maps a true source to the correct target using the ids of predicted pairs
        source2target = {
                        self._j2i.get(j, "NONE"): self._j2i.get(jt, "NONE")
                        for j,jt in zip(j_jt["seg_id"], j_jt["target_id"])
                        }

        if "NONE" in source2target:
            source2target.pop("NONE")


        if not source2target:
            pair_df["true_link"] = False
            return
        
        pair_df["true_link"] = pair_df.apply(check_true_pair, axis = 1, args = (source2target, ))


    def __create_pair_df(self, df: pd.DataFrame, pred :bool):


        def set_id_fn():
            pair_dict = dict()

            def set_id(row):
                p = tuple(sorted((row["p1"], row["p2"])))

                if p not in pair_dict:
                    pair_dict[p] = len(pair_dict)

                return pair_dict[p]

            return set_id

        
        first_df = df.groupby("seg_id", sort=False).first()
        first_df.reset_index(inplace=True)

        last_df = df.groupby("seg_id", sort=False).last()
        last_df.reset_index(inplace=True)



        if pred:
            first_target_df = self._df.groupby("seg_id", sort=False).first()
            j2link_label = {j:row["link_label"] for j, row in first_target_df.iterrows()}
            link_labels = [-1 if i not in self._i2j else j2link_label.get(self._i2j[i], -1) for i in first_df.index.to_numpy()]
            first_df["link_label"] = link_labels


        # we create ids for each memeber of the pairs
        # the segments in the batch will have unique ids starting from 0 to 
        # the total mumber of segments
        p1, p2 = [], []
        j = 0
        for _, gdf in df.groupby(level = 0, sort = False):
            n = len(gdf.loc[:, "seg_id"].dropna().unique())
            sample_seg_ids = np.arange(
                                        start= j,
                                        stop = j+n
                                        )
            p1.extend(np.repeat(sample_seg_ids, n).astype(int))
            p2.extend(np.tile(sample_seg_ids, n))
            j += n
    
        # setup pairs
        pair_df = pd.DataFrame({
                                "p1": p1,
                                "p2": p2,
                                })
                                

        if not len(pair_df.index):
            return pd.DataFrame()


        # create ids for each NON-directional pair
        pair_df["id"] = pair_df.apply(set_id_fn(), axis=1)

        #set the sample id for each pair
        pair_df["sample_id"] = first_df.loc[pair_df["p1"], "sample_id"].to_numpy()

        #set true the link_label
        #pair_df["link_label"] = first_df.loc[pair_df["p1"], "link_label"].to_numpy()

        #set start and end token indexes for p1 and p2
        pair_df["p1_start"] = first_df.loc[pair_df["p1"], "sample_token_id"].to_numpy()
        pair_df["p1_end"] = last_df.loc[pair_df["p1"], "sample_token_id"].to_numpy()

        pair_df["p2_start"] = first_df.loc[pair_df["p2"], "sample_token_id"].to_numpy()
        pair_df["p2_end"] = last_df.loc[pair_df["p2"], "sample_token_id"].to_numpy()

        # set directions
        pair_df["direction"] = 0  #self
        pair_df.loc[pair_df["p1"] < pair_df["p2"], "direction"] = 1 # ->
        pair_df.loc[pair_df["p1"] > pair_df["p2"], "direction"] = 2 # <-


        # mask for where p1 is a source        
        p1_source_mask = np.logical_or(pair_df["direction"] == 0 , pair_df["direction"] == 1)
        pair_df.loc[p1_source_mask, "link_label"] = first_df.loc[pair_df.loc[p1_source_mask, "p1"], "link_label"].to_numpy()

        #where p2 is a source
        p2_source_mask = pair_df["direction"] == 2
        pair_df.loc[p2_source_mask, "link_label"] = first_df.loc[pair_df.loc[p2_source_mask, "p2"], "link_label"].to_numpy()


        self.__add_link_matching_info(pair_df)


        if pred:
            pair_df["p1-ratio"] = pair_df["p1"].map(self._i2ratio)
            pair_df["p2-ratio"] = pair_df["p2"].map(self._i2ratio)
        else:
            pair_df["p1-ratio"] = 1
            pair_df["p2-ratio"] = 1

        return pair_df


    def __get_df_data(self,
                    level : str, 
                    key : str, 
                    flat : bool = False, 
                    pred : bool = False,
                    ) -> Union[Tensor, list, np.ndarray]:


        df = self._pred_df if pred else self._df

    
        if key == "lengths":
            data =  self.__get_lengths(df, level)

        # elif key == "lengths_tok":
        #     data = self.__seg_tok_lengths(df, level)

        elif key == "embs":
            data =  self.__get_pretrained_embeddings(df, level, flat = flat)

        elif key == "mask":
            data = self.__get_mask(level, pred = pred)

        else:
            if key == "span_idxs":
                data = self.__get_span_idxs(df, level)
            else:
                data = self.__get_column_values(df, level, key)

            if len(data) == 0:
                return data

            if isinstance(data[0], str):
                return data

            if not flat:

                if level == "am" and key == "span_idxs":
                    level = "adu"

                lengths = ensure_list(self.get(level, "lengths", pred = pred))

                data =  pad_sequence(
                                    torch.split(
                                                data, 
                                                lengths
                                                ), 
                                    batch_first = True,
                                    padding_value = -1 if self._task_regexp.search(key) else 0,
                                    )
        
        return data


    def __get_pair_df_data(self,
                    key : str, 
                    bidir : bool = True,   
                    ) -> Union[Tensor, list, np.ndarray]:


        if not hasattr(self, "_pair_df"):

            pred = not self.use_target_segs

            self._pair_df = self.__create_pair_df(
                                                df = self._pred_df if pred else self._df,
                                                pred = pred
                                                )

        pair_df = self._pair_df

        if not len(self._pair_df.index):
            return []

        if not bidir:
            pair_df = pair_df[pair_df["direction"].isin([0,1]).to_numpy()]

        if key == "lengths":
            data = pair_df.groupby("sample_id", sort=False).size().to_list()

        else:
            data = torch.LongTensor(pair_df[key].to_numpy())

        return data


    def __add_overlap_info(self):

        # we also have information about whether the seg_id is a true segments 
        # and if so, which TRUE segmentent id it overlaps with, and how much
        i2ratio, j2ratio, i2j, j2i = find_overlap(
                                                target_df = self._df,  
                                                pred_df = self._pred_df
                                                )

        #hacky temporary solution
        self._i2ratio = i2ratio
        self._i2j = i2j
        self._j2i = j2i

        # adding matching info to _df 
        self._df["i"] = self._df["seg_id"].map(j2i)
        self._df["i_ratio"] = self._df["seg_id"].map(j2ratio)

        # adding matching info to pred_df 
        self._pred_df["j"] = self._pred_df["seg_id"].map(i2j)
        self._pred_df["j_ratio"] = self._pred_df["seg_id"].map(i2ratio)

    
    @__sampling_wrapper
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

        # for level == pair We only have one pair_df as we are using the predicted or TARGET segments
        # to create candidate pairs
        # For other levels we have seperate dfs for TARGET and PREDICTIONS
        if level == "pair":
            data = self.__get_pair_df_data(
                                    key = key, 
                                    bidir = bidir,
                                    )
        else:
            
            data = self.__get_df_data(
                                    level = level, 
                                    key = key, 
                                    flat = flat, 
                                    pred = pred,
                                    )


        if isinstance(data, Tensor):
            data = data.to(self.device)

        return data


    def add(self, level:str, key:str, value:str):

        if key not in self.label_encoder.task_labels:
            raise KeyError(f"cannot add values to key ='{key}'")

        # if we are using TARGET segmentation results we  overwrite the 
        # columns of seg_id with TARGET seg_id as well as TARGET labels for each
        # task done in segmenation
        if "seg" in key and self.use_target_segs:

            self._pred_df["seg_id"] = self._df["seg_id"].to_numpy()

            for subtask in key.split("+"):
                self._pred_df[subtask] = self._df[subtask].to_numpy()

            self.__add_overlap_info()
            return
            
        
        if level == "token":
            mask = ensure_numpy(self.get("token", "mask")).astype(bool)
            self._pred_df.loc[:, key] = ensure_numpy(value)[mask]


        elif level == "seg":
            mask = ensure_numpy(self.get("seg", "mask")).astype(bool)
            seg_preds = ensure_numpy(value)[mask]

            # get the length of tokens for each seg 
            tok_lens = self._pred_df.groupby("seg_id", sort=False).size().to_numpy()
            
            # we spread the predictions on segments over tokens in TARGET segments
            cond = ~self._pred_df["seg_id"].isna()

            # expand the segment prediction for all their tokens 
            token_preds = np.repeat(seg_preds, tok_lens)
            
            #set the predictions for all rows which belong to a TARGET segment
            self._pred_df.loc[cond, key] = token_preds


        elif level == "p_seg":

            #get the lengths of each segment
            seg_lengths = self._pred_df.groupby("seg_id", sort=False).size().to_numpy()
            
            #expand the predictions over the tokens in the segments
            token_preds = np.repeat(value, seg_lengths)

            # as predicts are given in seg ids ordered from 0 to nr predicted segments
            # we can just remove all rows which doesnt belong to a predicted segments and 
            # it will match all the token preds and be in the correct order.
            self._pred_df.loc[~self._pred_df["seg_id"].isna(), key] = token_preds


        self._pred_df = self.label_encoder.validate(
                                                    task = key,
                                                    df = self._pred_df,
                                                    level = level,
                                                    )
        

        if "seg" in key:
            self.__add_overlap_info()


        # creating target_ids for links
        if key == "link":
            
            for si, sample_df in self._pred_df.groupby("sample_id", sort = False):
                
                # remove samples that doens have segments if we are predicting on segments
                segs = sample_df.groupby("seg_id", sort = False)
            
                # it might be helpfult to keep track on the global seg_id of the target
                # i.e. the seg_id of the linked segment
                seg_first = segs.first()

                links = seg_first["link"].to_numpy(dtype=int)

                target_ids = seg_first.index.to_numpy()[links]

                # remove rows outside segments
                is_not_nan = ~sample_df["seg_id"].isna()

                # exapnd target_id over the rows
                self._pred_df.loc[si].loc[is_not_nan, "target_id"] = np.repeat(target_ids, segs.size().to_numpy())
        

    def to(self, device):
        self.device = device

        # for k, level_class in self.items():
        #     level_class.device = device

