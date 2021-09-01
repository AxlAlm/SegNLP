

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
                seg_decoder = None,
                seg_gts_k: int = None,
                ):

        self.label_encoder = label_encoder

        # if we want to use ground truth in segmentation during training we can use
        # the following variable value to based on epoch use ground truth segmentation
        if seg_gts_k is not None:
            self.seg_gts = ScheduleSampling(
                                            schedule="inverse_sig",
                                            k=seg_gts_k
                                            )

        if seg_decoder is not None:
            self.seg_decoder = seg_decoder
        

    def __decode_segs(self):
    
        # we get the sample start indexes from sample lengths. We need this to tell de decoder where samples start
        sample_sizes = ensure_numpy(self.batch["token"]["lengths"]) #self.df.groupby(level=0, sort=False).size().to_numpy()
        sample_end_idxs = np.cumsum(sample_sizes)
        sample_start_idxs = np.concatenate((np.zeros(1), sample_end_idxs))[:-1]

        self.df["PRED", "seg_id"] = self.seg_decoder(
                                                self.df["PRED", "seg"].to_numpy(), 
                                                sample_start_idxs=sample_start_idxs.astype(int)
                                                )


    def __correct_links(self):
        """
        This function perform correction 3 mentioned in https://arxiv.org/pdf/1704.06104.pdf  (Appendix)
        Any link that is outside of the actuall text, e.g. when predicted link > max_idx, is set to predicted_link== max_idx
        """

        max_segs = self.df["PRED"].groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()
        self.df.loc["PRED", "max_seg"] = np.repeat(max_segs, self.df["PRED"].groupby(level=0, sort=False).size().to_numpy())

        above = self.df.loc["PRED", "link"] > self.df["PRED", "max_seg"]
        below =self.df.loc["PRED", "link"] < 0

        self.df["PRED"].loc[above | below, "link"] = self.df["PRED"].loc[above | below, "max_seg"]


    def __ensure_homogeneous(self, subtask):

        """
        ensures that the labels inside a segments are the same. For each segment we take the majority label 
        and use it for the whole span.
        """
        df = self.df.loc["PRED",["seg_id", subtask]].value_counts(sort=False).to_frame()
        df.reset_index(inplace=True)
        df.rename(columns={0:"counts"}, inplace=True)
        df.drop_duplicates(subset=['seg_id'], inplace=True)

        seg_lengths = self.df["PRED"].groupby("seg_id", sort=False).size()
        most_common = np.repeat(df[subtask].to_numpy(), seg_lengths)

        self.df["PRED"].loc[~self.df["PRED", "seg_id"].isna(), subtask] = most_common


    def step(self, batch: BatchInput, step_type : str):

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
        try:
            self.__use_gt_seg = getattr(self, "seg_gts")(self.batch.current_epoch) and step_type == "train"
        except AttributeError:
            self.__use_gt_seg = False

        return self


    def add_stuff(self, stuff):
        self.stuff.update(stuff)


    def add_logits(self, logits:Tensor, task:str):
        self.logits[task] = logits


    def add_preds(self, preds:Union[np.ndarray, Tensor], level:str,  task:str):
        
        # if we are using the segmentation ground truths we overwrite the segmentation labels
        # aswell as segment ids
        if self.__use_gt_seg and "seg" in task:
            
            for subtask in task.split("+"):
                self.loc["PRED", subtask] = self.loc["TARGET", subtask].to_numpy()

            self.loc["PRED", "seg_id"] = self.loc["TARGET", "seg_id"].to_numpy()
            return
            


        if level == "token":
            mask = ensure_numpy(self.batch["token"]["mask"]).astype(bool)
            self.df["PRED", task] = ensure_numpy(preds)[mask]
        

        elif level == "seg":
            mask = ensure_numpy(self.batch["seg"]["mask"]).astype(bool)
            seg_preds = ensure_numpy(preds)[mask]
            
            # we spread the predictions on segments over all tokens in the segments
            cond = ~self.df["PRED", "seg_id"].isna()

            # repeat the segment prediction for all their tokens 
            token_preds = np.repeat(seg_preds, ensure_numpy(self.batch["seg"]["lengths_tok"])[mask])

            self.df["PRED"].loc[cond, task] = token_preds


        elif level == "p_seg":
            seg_tok_lengths = self.df["PRED"].groupby("seg_id", sort=False).size()
            token_preds = np.repeat(preds, seg_tok_lengths)

            # as predicts are given in seg ids ordered from 0 to nr predicted segments
            # we can just remove all rows which doesnt belong to a predicted segments and 
            # it will match all the token preds and be in the correct order.
            self.df["PRED"].loc[~self.df["PRED", "seg_id"].isna(), task] = token_preds


        subtasks = task.split("+")  

        #make sure we do segmentation first
        if "seg" in subtasks:
            subtasks.remove("seg")
            self.__decode_segs()

        if len(subtasks) > 1:
            # break tasks such as seg+label into seg and label, e.g. 1 -> I + Premise -> [1, 2]
            self.df["PRED"] = self.label_encoder.decouple(
                                        task = task, 
                                        subtasks = subtasks, 
                                        df = self.df["PRED"], 
                                        level = level
                                        )

        for subtask in subtasks:

            if level == "token":
                self.__ensure_homogeneous(subtask)
        
    
            if subtask == "link":
                # if level is segment and we are not using a ground truth segment sampler  we can correct links
                # based on the ground truth segments else we will correct based on the predicted segments
                #true_segs = level == "seg" and not hasattr(self, "seg_gts")
                self.__correct_links()    


    @utils.Memorize
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


        df = self.df["TARGET" if self.__use_gt else "PRED"]
        
        first_df = df.groupby("seg_id", sort=False).first()
        first_df.reset_index(inplace=True)

        last_df = df.groupby("seg_id", sort=False).last()
        last_df.reset_index(inplace=True)

        # we create ids for each memeber of the pairs
        # the segments in the batch will have unique ids starting from 0 to 
        # the total mumber of segments
        p1, p2 = [], []
        j = 0
        for i in range(len(self.batch)):
            n = len(df.loc[i,"seg_id"].dropna().unique())
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
        if not self.__use_seg_gt:

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
        false_linked_pairs = first_df.iloc[normalized_links].index.to_numpy() == p2

        # creating a mask over all pairs which tells us which pairs include a segmnets
        # which is not a TRUE segment, i.e. overlaps with a ground truth segments to a certain
        # configurable extend.
        #if self.threshohold == "first":
        #else:
        p1_cond = pair_df["p1-ratio"] >= 0.5
        p2_cond = pair_df["p2-ratio"] >= 0.5
        contain_false_segs = np.logical_and(p1_cond.to_numpy(), p2_cond.to_numpy())
    
        pair_df["false_pairs"] = np.logical_and(false_linked_pairs, contain_false_segs)

        nodir_pair_df = pair_df[pair_df["direction"].isin([0,1]).to_numpy()]

        pair_dict = {
                    "bidir": {k:torch.tensor(v, device=self.batch.device) for k,v in pair_df.to_dict("list").items()},
                    "nodir": {k:torch.tensor(v, device=self.batch.device) for k,v in nodir_pair_df.to_dict("list").items()}
                    }

        pair_dict["bidir"]["lengths"] = pair_df.groupby("sample_id", sort=False).size().to_list()
        pair_dict["nodir"]["lengths"] = nodir_pair_df.groupby("sample_id", sort=False).size().to_list()

        return pair_dict


    @utils.Memorize
    def get_seg_data(self):


        if self.__use_gt_seg:
            seg_data = {
                            "span_idxs": self.batch["seg"]["span_idxs"],
                            "lengths": self.batch["seg"]["lengths"],
                            }   
        else:

            df = self.df["PRED"]


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
        pass
        
        # task_key = task

        # # if we want to use schedule sampling we select the ground truths instead of 
        # # the predictions
        # if self.__use_seg_gt:
        #     task_key = f"T-{task}"

        # # we take the lengths in tokens for each sample
        # tok_lengths = self.df.groupby(level=0, sort=False).size().to_numpy()

        # # create the end indexes. Remove the last value as its the end and will create a faulty split
        # end_idxs = np.cumsum(tok_lengths)[:-1]

        # # split and then pad
        # preds = utils.zero_pad(np.hsplit(self.df[task_key].to_numpy(), end_idxs))

        # #then we create one_hots from the preds if thats needed
        # if one_hot:
        #     return utils.one_hot(
        #                             matrix = torch.LongTensor(preds),
        #                             mask = self.batch["token"]["mask"],
        #                             num_classes = len(self.label_encoders[task])
        #                             )
        # else:
        #return preds
        







    # def extract_match_info(df):
        

    #     def overlap(target, pdf):
    #         j = target["T-seg_id"].to_list()[0]
    #         seg_ids = target["seg_id"].dropna().to_list()

    #         if not seg_ids:
    #             return [0, 0, -1, -1]

    #         i = Counter(seg_ids).most_common(1)[0][0]

    #         p_index = set(pdf.loc[[i], "index"]) #slowest part
    #         t_index = set(target.index)

    #         ratio = len(t_index.intersection(p_index)) / max(len(p_index), len(t_index))

    #         return ratio, i, j


    #     # create pdf with predicted segments ids as index to be able
    #     # to select rows faster
    #     pdf = df.copy()
    #     pdf["index"] = pdf.index 
    #     pdf.index = pdf["seg_id"]

    #     # we extract matching information. Which predicted segments are overlapping with which 
    #     # ground truth segments
    #     match_info = np.vstack(df.groupby("T-seg_id").apply(overlap, (pdf)))
    #     ratio = match_info[:,0]
    #     i = match_info[:,2] #predicted segment id
    #     j = match_info[:,3] # ground truth segment id
        
    #     # #contains mapping between i j where i is an exact/approx match for j
    #     # i2j_exact = dict(zip(i[exact],j[exact]))
    #     # i2j_approx = dict(zip(i[approx],j[approx]))

    #     return exact, approx, i2j_exact, i2j_approx, i






    # def _add_segs(self, ):

    #     if level == "seg":

    #         span_indexes = ensure_numpy(self.batch["seg"]["span_idxs"])
    #         data = self.__unfold_span_labels(
    #                                         span_labels=data,
    #                                         span_indexes=span_indexes,
    #                                         max_nr_token=max(ensure_numpy(self.batch["token"]["lengths"])),
    #                                         fill_value= 0 if task == "link" else -1
    #                                         )

    #         if not self._pred_span_set:
    #             self.pred_seg_info["seg"]["lengths"] = self.batch["seg"]["lengths"]
    #             self.pred_seg_info["span"]["lengths"] = self.batch["span"]["lengths"]
    #             self.pred_seg_info["span"]["lengths_tok"] = self.batch["span"]["lengths_tok"]
    #             self.pred_seg_info["span"]["none_span_mask"] = self.batch["span"]["none_span_mask"]
    #             self._pred_span_set = True

    #         level = "token"


    # def __add_to_df(self, data, level, task):

    #     if level == "seg":
    #         data = self.__unfold_span_labels(
    #                                         span_labels=data,
    #                                         span_indexes=ensure_numpy(self.batch["seg"]["span_idxs"]),
    #                                         max_nr_token=max(ensure_numpy(self.batch["token"]["lengths"])),
    #                                         fill_value= 0 if task == "link" else -1
    #                                         )

    #     mask = ensure_numpy(self.batch[level]["mask"])
    #     preds_flat = ensure_flat(ensure_numpy(preds), mask=mask)
    #     targets_flat = ensure_flat(ensure_numpy(targets), mask=mask)

    #     self._outputs[task] = preds_flat
    #     self._outputs[f"T-{task}"] = targets_flat


 




    # def add_preds(self, task:str, level:str, data:torch.tensor, decoded:bool=False):

    #     #assert task in set(self.tasks), f"{task} is not a supported task. Supported tasks are: {self.tasks}"
    #     assert level in set(["token", "seg"]), f"{level} is not a supported level. Only 'token' or 'seg' are supported levels"
    #     assert torch.is_tensor(data) or isinstance(data, np.ndarray), f"{task} preds need to be a tensor or numpy.ndarray"
    #     assert len(data.shape) == 2, f"{task} preds need to be a 2D tensor"

    #     data = ensure_numpy(data)

    
    #     if level == "seg":
    #         if not self._pred_span_set:
    #             self.pred_seg_info["seg"]["lengths"] = self.batch["seg"]["lengths"]
    #             self.pred_seg_info["span"]["lengths"] = self.batch["span"]["lengths"]
    #             self.pred_seg_info["span"]["lengths_tok"] = self.batch["span"]["lengths_tok"]
    #             self.pred_seg_info["span"]["none_span_mask"] = self.batch["span"]["none_span_mask"]
    #             self._pred_span_set = True


    #     if "+" in task:
    #         self.__handle_complex_tasks(
    #                                     data=data,
    #                                     level=level,
    #                                     lengths=ensure_numpy(self.batch[level]["lengths"]),
    #                                     task=task
    #                                     )
    #         return




        # if level == "seg":

        #     span_indexes = ensure_numpy(self.batch["seg"]["span_idxs"])
        #     data = self.__unfold_span_labels(
        #                                     span_labels=data,
        #                                     span_indexes=ensure_numpy(self.batch["seg"]["span_idxs"]),
        #                                     max_nr_token=max(ensure_numpy(self.batch["token"]["lengths"])),
        #                                     fill_value= 0 if task == "link" else -1
        #                                     )

        #     if not self._pred_span_set:
        #         self.pred_seg_info["seg"]["lengths"] = self.batch["seg"]["lengths"]
        #         self.pred_seg_info["span"]["lengths"] = self.batch["span"]["lengths"]
        #         self.pred_seg_info["span"]["lengths_tok"] = self.batch["span"]["lengths_tok"]
        #         self.pred_seg_info["span"]["none_span_mask"] = self.batch["span"]["none_span_mask"]
        #         self._pred_span_set = True

        #     level = "token"


        # if task == "link":
        #     preds = self.__correct_token_links(
        #                                                 data,
        #                                                 lengths_segs=ensure_numpy(self.pred_seg_info["seg"]["lengths"]),
        #                                                 span_token_lengths=ensure_numpy(self.pred_seg_info["span"]["lengths_tok"]),
        #                                                 none_spans=ensure_numpy(self.pred_seg_info["span"]["none_span_mask"]),
        #                                                 decoded=decoded,
        #                                                 )
     

        # if task == "seg":

        #     self.pred_seg_info["seg"]["lengths"] = seg_data["seg"]["lengths"]
        #     self.pred_seg_info["span"]["lengths"] = seg_data["span"]["lengths"]
        #     self.pred_seg_info["span"]["lengths_tok"] = seg_data["span"]["lengths_tok"]
        #     self.pred_seg_info["span"]["none_span_mask"] = seg_data["span"]["none_span_mask"]
            

        #     token_seg_ids, token_span_ids = self.__get_token_seg_ids(
        #                                                     span_token_lengths = self.pred_seg_info["span"]["lengths_tok"],
        #                                                     none_span_masks = self.pred_seg_info["span"]["none_span_mask"],
        #                                                     seg_lengths = self.pred_seg_info["seg"]["lengths"],
        #                                                     span_lengths = self.pred_seg_info["span"]["lengths"],
        #                                                     )
        #     self._outputs["seg_id"] = token_seg_ids
        #     self._outputs["span_id"] = token_span_ids



    #     mask = ensure_numpy(self.batch[level]["mask"])
    #     preds_flat = ensure_flat(ensure_numpy(preds), mask=mask)
    #     targets_flat = ensure_flat(ensure_numpy(targets), mask=mask)

    #     self._outputs[task] = preds_flat
    #     self._outputs[f"T-{task}"] = targets_flat


    # def __init__output(self, batch):

    #     # INIT
    #     self.batch = batch
    #     self._outputs = {}
    #     self.pred_seg_info = {
    #                             "seg":{},
    #                             "span":{},
    #                             }
    #     self._pred_span_set = False
    #     self.max_tok = int(torch.max(batch["token"]["lengths"], dim=-1).values)
    #     self.max_seg = int(torch.max(batch["seg"]["lengths"], dim=-1).values)


    #     self._outputs["sample_id"] = np.concatenate([np.full(int(l),int(i)) for l,i in zip(self.batch["token"]["lengths"],self.batch.ids)])
    #     self._outputs["text"] = ensure_flat(ensure_numpy(self.batch["token"]["text"]), mask=ensure_numpy(self.batch["token"]["mask"]))

    #     token_ids = [self.batch["token"]["token_ids"][i,:self.batch["token"]["lengths"][i]] for i in range(len(self.batch))]
    #     self._outputs["token_id"] = ensure_numpy([int(t) for sub in token_ids for t in sub])

    #     if not self.inference:  
    #         token_seg_ids, token_span_ids, = self.__get_token_seg_ids(
    #                                                             span_token_lengths = self.batch["span"]["lengths_tok"],
    #                                                             none_span_masks = self.batch["span"]["none_span_mask"],
    #                                                             seg_lengths = self.batch["seg"]["lengths"],
    #                                                             span_lengths = self.batch["span"]["lengths"],
    #                                                         )

    #         self._outputs["T-seg_id"] = token_seg_ids
    #         self._outputs["seg_id"] = token_seg_ids

    #         self._outputs["T-span_id"] = token_span_ids
    #         self._outputs["span_id"] = token_span_ids


    # #@utils.timer
    # def format(self, batch:dict, preds:dict):

        # self.__init__output(batch)

        # for task, data in preds.items():

        #     if data.shape[-1] == self.max_seg:
        #         level = "seg"
        #     elif data.shape[-1] == self.max_tok:
        #         level = "token"
        #     else:
        #         raise RuntimeError(f"data was given in shape {data.shape[0]} but {self.max_seg} (segment level) or {self.max_tok} (token level) was expected")

        #     self.__add_preds(
        #                     task=task, 
        #                     level=level,
        #                     data=data,
        #                     )
         
        # return pd.DataFrame(self._outputs)