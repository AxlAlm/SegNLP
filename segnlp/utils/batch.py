
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

class Batch:


    def __init__(self, 
                df: pd.DataFrame, 
                pretrained_features: dict = {},
                device = None
                ):
        self._df = df
        self._pred_df = df.copy(deep=True)
        self._task_regexp = re.compile("seg|link|label|link_label")
        self._pretrained_features = pretrained_features
        self.device = device
        self.__ok_levels = set(["seg", "token", "span"])

        if "am_id" in self._df.columns:
            self.__ok_levels.update(["am_id", "adu_id"])

        self._size = self._df["sample_id"].nunique()


    def __len__(self):
        return self._size
 

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


    def __get_mask(self, df: pd.DataFrame, level:str, ):
        return utils.create_mask(self.get(df, level, "lengths"), as_bool = True)


    def __get_lengths(self, df: pd.DataFrame, level:str):

        if level == "token":
            return torch.LongTensor(df.groupby(level=0, sort = False).size().to_numpy())
        else:
            return torch.LongTensor(df.groupby(level=0, sort=False)[f"{level}_id"].nunique().to_numpy())


    def __get_pretrained_embeddings(self, level:str, flat:bool):

        if level == "token":
            embs = self._pretrained_features["word_embs"]
        else:
            embs = self._pretrained_features["seg_embs"]

        embs = embs[:, :max(self._get_lengths(level)), :]

        if flat:
            embs = embs[self._get_mask("level")]

        return torch.tensor(embs, dtype = torch.float)


    def __create_pair_df(self, pred : bool = False):


        def set_id_fn():
            pair_dict = dict()

            def set_id(row):
                p = tuple(sorted((row["p1"], row["p2"])))

                if p not in pair_dict:
                    pair_dict[p] = len(pair_dict)
                
                return pair_dict[p]

            return set_id


        df = self._pred_df if pred else self._df

        first_df = df.groupby("seg_id", sort=False).first()
        first_df.reset_index(inplace=True)

        last_df = df.groupby("seg_id", sort=False).last()
        last_df.reset_index(inplace=True)


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
        
        # create ids for each NON-directional pair
        pair_df["id"] = pair_df.apply(set_id_fn(), axis=1)

        # #set the sample id for each pair
        # pair_df["sample_id"] = first_df.loc[pair_df["p1"], "sample_id"].to_numpy()

        #set true the link_label
        pair_df["link_label"] = first_df.loc[pair_df["p1"], "link_label"].to_numpy()

        #set start and end token indexes for p1 and p2
        pair_df["p1_start"] = first_df.loc[pair_df["p1"], "sample_token_id"].to_numpy()
        pair_df["p1_end"] = last_df.loc[pair_df["p1"], "sample_token_id"].to_numpy()

        pair_df["p2_start"] = first_df.loc[pair_df["p2"], "sample_token_id"].to_numpy()
        pair_df["p2_end"] = last_df.loc[pair_df["p2"], "sample_token_id"].to_numpy()

        # set directions
        pair_df["direction"] = 0  #self
        pair_df.loc[pair_df["p1"] < pair_df["p2"], "direction"] = 1 # ->
        pair_df.loc[pair_df["p1"] > pair_df["p2"], "direction"] = 2 # <-

        # finding the matches between predicted segments and true segments
        if pred:

            # we also have information about whether the seg_id is a true segments 
            # and if so, which TRUE segmentent id it overlaps with, and how much
            seg_id, T_seg_id, ratio = overlap_ratio(
                                                    target_df = self.df.loc["TARGET"],  
                                                    pred_df = self.df.loc["PRED"]
                                                    )

            p1_matches = np.isin(pair_df["p1"], seg_id)
            p2_matches = np.isin(pair_df["p2"], seg_id)




            # adding true seg ids for each p1,p2
            i2j = dict(zip(seg_id, T_seg_id))

            p1_v = np.array(p1, dtype=np.float)
            p1_v[~p1_matches] = np.nan

            p2_v = np.array(p2, dtype=np.float)
            p2_v[~p2_matches] =  np.nan

            pair_df["p1"] = p1_v
            pair_df["p2"] = p2_v
            pair_df["p1"] = pair_df["p1"].map(i2j)
            pair_df["p2"] = pair_df["p2"].map(i2j)

            # adding ratio for true seg ids for each p1,p2
            i2ratio = dict(zip(seg_id, ratio))

            p1_ratio_default = np.array(p1, dtype=np.float)
            p1_ratio_default[~p1_matches] = float("-inf")

            p2_ratio_default = np.array(p2, dtype=np.float)
            p2_ratio_default[~p2_matches] = float("-inf")

            pair_df["p1-ratio"] = p1_ratio_default
            pair_df["p2-ratio"] = p2_ratio_default
            pair_df["p1-ratio"] = pair_df["p1-ratio"].map(i2ratio)
            pair_df["p2-ratio"] = pair_df["p2-ratio"].map(i2ratio)
        
        else:
            pair_df["p1"] = p1
            pair_df["p2"] = p2
            pair_df["p1-ratio"] = 1
            pair_df["p2-ratio"] = 1
        

        # We also need to create mask which tells us which pairs either:
        # 1; include NON-LINKING segments
        # 2; include segments which do not match/overlap sufficiently with a 
        # ground truth segment

        # 1 find which pairs are "false", i.e. the members whould not be linked
        links = first_df.loc[pair_df["p1"], "link"].to_numpy()
        pairs_per_sample = pair_df.groupby("sample_id", sort=False).size().to_numpy()
        seg_per_sample = utils.np_cumsum_zero(first_df.groupby("sample_id", sort=False).size().to_numpy())
        normalized_links  = links + np.repeat(seg_per_sample, pairs_per_sample)
        pair_df["true_link"] = first_df.iloc[normalized_links].index.to_numpy() == p2


        nodir_pair_df = pair_df[pair_df["direction"].isin([0,1]).to_numpy()]





        pair_dict = {
                    "bidir": {k:torch.tensor(v, device=self.batch.device) for k,v in pair_df.to_dict("list").items()},
                    "nodir": {k:torch.tensor(v, device=self.batch.device) for k,v in nodir_pair_df.to_dict("list").items()}
                    }

        pair_dict["bidir"]["lengths"] = pair_df.groupby("sample_id", sort=False).size().to_list()
        pair_dict["nodir"]["lengths"] = nodir_pair_df.groupby("sample_id", sort=False).size().to_list()


        lens = nodir_pair_df.groupby("sample_id", sort=False).size().to_list()


        starts = torch.split(torch.LongTensor(nodir_pair_df["p1_start"].to_numpy()), lens)
        ends  = torch.split(torch.LongTensor(nodir_pair_df["p2_end"].to_numpy()), lens)

        return pair_dict


    @utils.Memorize
    def get(self, 
            level:str, 
            key:str, 
            flat:bool = False, 
            pred : bool = False,
            bidir : bool = True
            ):

        if level not in self.__ok_levels:
            raise KeyError

        
        if level == "pair" and hasattr(self, f"_pair_df{'_pred' if pred else ''}"):
            self.__create_pair_df()


        df = self._pred_df if pred else self._df
    
        if key == "lengths":
            data =  self.__get_lengths(df, level)

        elif key == "embs":
            data =  self.__get_pretrained_embeddings(df, level, flat=flat)

        elif key == "mask":
            data = self.__get_mask(df, level)

        else:
            if key == "span_idxs":
                data = self.__get_span_idxs(df, level)
            else:
                data = self.__get_column_values(df, level, key)


            if isinstance(data[0], str):
                return data

            if not flat:

                if level == "am" and key == "span_idxs":
                    level = "adu"
            
                lengths = utils.ensure_list(self.get(df, level, "lengths"))
                    
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


    def add(self, level:str, key:str, value:str):
        
        if level == "token":
            mask = ensure_numpy(self.get("token", "mask")).astype(bool)
            self._pred_df.loc[:, key] = ensure_numpy(value)[mask]


        elif level == "seg":
            mask = ensure_numpy(self.get("seg", "mask")).astype(bool)
            seg_preds = ensure_numpy(value)[mask]
            
            # we spread the predictions on segments over all tokens in the segments
            cond = ~self._pred_df.loc["seg_id"].isna()

            # repeat the segment prediction for all their tokens 
            token_preds = np.repeat(seg_preds, ensure_numpy(self.get("seg", "lengths_tok"))[mask])

            self._pred_df.loc.loc[cond, key] = token_preds


        elif level == "p_seg":
            seg_tok_lengths = self._pred_df.loc.groupby("seg_id", sort=False).size().to_numpy()
            
        
            token_preds = np.repeat(value, seg_tok_lengths)

            # as predicts are given in seg ids ordered from 0 to nr predicted segments
            # we can just remove all rows which doesnt belong to a predicted segments and 
            # it will match all the token preds and be in the correct order.
            self._pred_df.loc[~self._pred_df.loc["seg_id"].isna(), key] = token_preds


        self._pred_df = self.label_encoder.validate(
                                                    task = key,
                                                    df = self._pred_df,
                                                    level = level,
                                                    )
                

    def to(self, device):
        self.device = device

        # for k, level_class in self.items():
        #     level_class.device = device

