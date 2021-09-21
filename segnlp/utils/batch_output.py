

#basics
from inspect import getsource
from re import sub
from typing import Union, List, DefaultDict, Tuple
from unittest import result
from networkx.algorithms.assortativity import pairs
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import product
from itertools import combinations
from time import time
from functools import wraps
from functools import lru_cache

#pytorch
import torch
from torch import Tensor



#segnlp
from .array import ensure_flat, ensure_numpy, flatten
from .schedule_sample import ScheduleSampling
from segnlp import utils
from .batch_input import BatchInput
from .label_encoder import LabelEncoder


class BatchOutput:

    def __init__(self, 
                label_encoder : LabelEncoder,
                ):

        self.label_encoder = label_encoder

    @utils.timer
    def step(self, batch: BatchInput,  use_target_segs :bool = False):


        # create a copy of the original dataframe for all predictions
        pred_df = batch._df.copy(deep=True)
        for task in self.label_encoder.task_labels.keys():
            pred_df[task] = None  

        #we will use the TARGET segment ids if the prediction level is on segment level
        if  "seg" not in self.label_encoder.task_labels:
            pred_df["seg_id"] = batch._df["seg_id"].to_numpy()
        else:
            pred_df["seg_id"] = None
        

        # we made a multi-index dict to keep the TARGET and PRED labels seperate
        self.df  = pd.concat((batch._df, pred_df), keys= ['TARGET', 'PRED'], axis=0)

        self.stuff = {}
        self.logits = {}
        self.batch = batch

        # if we want to use schedule sampling we select the ground truths segmentation instead of 
        # the predictions. Only for Training steps
        self.__use_gt_seg = use_target_segs
     
        return self


    def add_stuff(self, stuff):
        self.stuff.update(stuff)


    def add_logits(self, logits:Tensor, task:str):
        self.logits[task] = logits


    def add_preds(self, preds:Union[np.ndarray, Tensor], level:str,  task:str):
    

        # if we are using the segmentation ground truths we overwrite the segmentation labels
        # aswell as segment ids
        if self.__use_gt_seg and "seg" in task:
            
            self.df.loc["PRED", task] = self.df.loc["TARGET", task].to_numpy()

            for subtask in task.split("+"):
                self.df.loc["PRED", subtask] = self.df.loc["TARGET", subtask].to_numpy()

            self.df.loc["PRED", "seg_id"] = self.df.loc["TARGET", "seg_id"].to_numpy()
            return
        

        if level == "token":
            mask = ensure_numpy(self.batch.get("token", "mask")).astype(bool)
            self.df.loc["PRED", task] = ensure_numpy(preds)[mask]


        elif level == "seg":
            mask = ensure_numpy(self.batch.get("seg", "mask")).astype(bool)
            seg_preds = ensure_numpy(preds)[mask]
            
            # we spread the predictions on segments over all tokens in the segments
            cond = ~self.df.loc["PRED", "seg_id"].isna()

            # repeat the segment prediction for all their tokens 
            token_preds = np.repeat(seg_preds, ensure_numpy(self.batch.get("seg", "lengths_tok"))[mask])

            self.df.loc["PRED"].loc[cond, task] = token_preds


        elif level == "p_seg":

            print("UNIQUE SEG IDS ADD_PREDs", self.df.loc["PRED","seg_id"].nunique())

            seg_tok_lengths = self.df.loc["PRED"].groupby("seg_id", sort=False).size().to_numpy()
            
            
            d  = len(self.df.loc["TARGET"].groupby("seg_id", sort=False).size().to_numpy())

            #print(preds, seg_tok_lengths)
            print(self.__use_gt_seg, len(preds), len(seg_tok_lengths), d)
            token_preds = np.repeat(preds, seg_tok_lengths)

            # as predicts are given in seg ids ordered from 0 to nr predicted segments
            # we can just remove all rows which doesnt belong to a predicted segments and 
            # it will match all the token preds and be in the correct order.
            self.df.loc["PRED"].loc[~self.df.loc["PRED", "seg_id"].isna(), task] = token_preds


        self.df.loc["PRED"] = self.label_encoder.validate(
                                                            task = task,
                                                            df = self.df.loc["PRED"].copy(deep = True),
                                                            level = level,
                                                            ).values
                

    #@utils.Memorize
    def get_pair_data(self):

        def set_id_fn():
            pair_dict = dict()

            def set_id(row):
                p = tuple(sorted((row["p1"], row["p2"])))

                if p not in pair_dict:
                    pair_dict[p] = len(pair_dict)
                
                return pair_dict[p]

            return set_id


        def extract_match_info(self, df):
            

            def overlap(target, pdf):
                j = target["seg_id"].to_list()[0]
                seg_ids = target["P-seg_id"].dropna().to_list()

                if not seg_ids:
                    return np.array([None, None, None])

                i = Counter(seg_ids).most_common(1)[0][0]

                p_index = set(pdf.loc[[i], "sample_token_id"]) #slowest part
                t_index = set(target["sample_token_id"])

                ratio = len(t_index.intersection(p_index)) / max(len(p_index), len(t_index))
                return np.array([i, j, ratio])

            # create pdf with predicted segments ids as index to be able
            # to select rows faster
            pdf = df.copy()
            pdf["index"] = pdf.index 
            pdf.index = pdf["seg_id"]

            # we extract matching information. Which predicted segments are overlapping with which 
            # ground truth segments
            match_info = np.vstack(df.groupby("seg_id", sort=False).apply(overlap, (pdf)))
            
            i = match_info[:,0].astype(int) #predicted segment id
            j = match_info[:,1].astype(int) # ground truth segment id
            ratio = match_info[:,2]

            return i, j, ratio


        df = self.df.loc["TARGET" if self.__use_gt_seg else "PRED"]


        print("UNIQUE SEG IDS", df.loc[:,"seg_id"].nunique())

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

        #set the sample id for each pair
        pair_df["sample_id"] = first_df.loc[pair_df["p1"], "sample_id"].to_numpy()

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
        if not self.__use_gt_seg:

            # we also have information about whether the seg_id is a true segments 
            # and if so, which TRUE segmentent id it overlaps with, and how much
            seg_id, T_seg_id, ratio = extract_match_info(df)

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


    #@utils.Memorize
    def get_seg_data(self):

        if self.__use_gt_seg:
            seg_data = {
                            "span_idxs": self.batch.get("seg", "span_idxs"),
                            "lengths": self.batch.get("seg", "lengths"),
                            }   
        else:

            df = self.df.loc["PRED"]

            seg_lengths = df[~df["seg_id"].isna()].groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()

            start = df.groupby("seg_id", sort=False).first()["sample_token_id"]
            end = df.groupby("seg_id",  sort=False).last()["sample_token_id"]

            span_idxs_flat = list(zip(start, end))

            assert len(span_idxs_flat) == np.sum(seg_lengths), f"{len(span_idxs_flat)} {np.sum(seg_lengths)}"

            span_idxs = np.zeros((len(self.batch), np.max(seg_lengths), 2))
            floor = 0
            for i,l in enumerate(seg_lengths):
                span_idxs[i][:l] = np.array(span_idxs_flat[floor:floor+l])
                floor += l

            seg_data = {
                        "span_idxs": span_idxs,
                        "lengths": seg_lengths
                        }   
        return seg_data



    def get_preds(self, task:str, one_hot:bool = False):
        
        # create the end indexes. Remove the last value as its the end and will create a faulty split
        end_idxs = np.cumsum(self.batch.get("token", "lengths"))[:-1]

        # split and then pad
        preds = utils.zero_pad(
                                np.hsplit(
                                        self.df.loc["PRED", task].to_numpy(), 
                                        end_idxs
                                        )
                            )

        #then we create one_hots from the preds if thats needed
        if one_hot:
            return utils.one_hot(
                                    matrix = torch.LongTensor(preds),
                                    mask = self.batch.get("token", "mask"),
                                    num_classes = len(self.label_encoder.task_labels[task])
                                    )
        else:
            return preds
