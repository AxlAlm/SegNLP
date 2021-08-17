

#basics
from inspect import getsource
from re import sub
from typing import Union, List, DefaultDict, Tuple
from unittest import result
from networkx.algorithms.assortativity import pairs
import numpy as np
from numpy.lib import utils
import pandas as pd
from collections import Counter, defaultdict
from itertools import product
from itertools import combinations
from time import time
from functools import wraps


#segnlp
from .input import Input
from .array import ensure_flat, ensure_numpy, flatten
from .bio_decoder import BIODecoder
from .schedule_sample import ScheduleSampling
from segnlp import utils

#pytorch
import torch
from torch import Tensor


class Output:

    def __init__(self, 
                label_encoders:dict,
                tasks:list,
                all_tasks:list,
                subtasks:list,
                prediction_level:str,
                inference:bool,
                sampling_k:int=0,
                ):

        self.inference = inference
        self.label_encoders = label_encoders
        self.tasks = tasks
        self.all_tasks = all_tasks
        self.prediction_level = prediction_level
        self.subtasks = subtasks

        self.schedule = ScheduleSampling(
                                        schedule="inverse_sig",
                                        k=sampling_k
                                        )

        seg_task = [task for task in self.tasks if "seg" in task]
        if seg_task:
            #id2label = label_encoders[seg_task[0]].id2label
            self.seg_decoder = BIODecoder()
                                        # B = [i for i,l in id2label.items() if "B-" in l],
                                        # I = [i for i,l in id2label.items() if "I-" in l],
                                        # O = [i for i,l in id2label.items() if "O-" in l],
                                        # )

        self.__decouplers = self.__create_decouplers(label_encoders, tasks)
        

    def _cache(f):
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            v_name =  f.__name__.split("_",1)[1]
            if hasattr(self, v_name):
                return getattr(self, v_name)
            else:
                return_value = f(self, *args, **kwargs)
                setattr(self, v_name, return_value)
                return return_value
        return wrapped


    def __create_decouplers(self, label_encoders, tasks):
        
        decouplers = {}
        for task in tasks:

            if "+" not in task:
                continue
            
            decouplers[task] = {}

            labels = label_encoders[task].label2id.keys()
            subtasks = task.split("+")

            for i,label in enumerate(labels):

                sublabels = label.split("_")
                decouplers[task][i] = [label_encoders[st].encode(sl) for st, sl in zip(subtasks, sublabels)]

        return decouplers
    

    def __decode_segs(self):
    
        # we get the sample start indexes from sample lengths. We need this to tell de decoder where samples start
        sample_sizes = self.df.groupby(level=0).size().to_numpy()
        sample_end_idxs = np.cumsum(sample_sizes)
        sample_start_idxs = np.concatenate((np.zeros(1), sample_end_idxs))[:-1]

        self.df["seg_id"] = self.seg_decoder(
                                                self.df["seg"].to_numpy(), 
                                                sample_start_idxs=sample_start_idxs
                                                )


    def __correct_links(self, true_segs:bool=False):
        """
        This function perform correction 3 mentioned in https://arxiv.org/pdf/1704.06104.pdf  (Appendix)
        Any link that is outside of the actuall text, e.g. when predicted link > max_idx, is set to predicted_link== max_idx
        """

        max_segs = self.df.groupby(level=0)["T-seg_id" if true_segs else "seg_id"].nunique().to_numpy()
        self.df["max_seg"] = np.repeat(max_segs, self.df.groupby(level=0).size().to_numpy())

        above = self.df["link"] > self.df["max_seg"]
        below = self.df["link"] < 0

        self.df.loc[above,"link"] = self.df.loc[above,"max_seg"]
        self.df.loc[below,"link"] = self.df.loc[below,"max_seg"]


    def __ensure_homogeneous(self, subtask):

        """
        ensures that the labels inside a segments are the same. For each segment we take the majority label 
        and use it for the whole span.
        """
        df = self.df.loc[:,["seg_id", subtask]].value_counts().to_frame()
        df.reset_index(inplace=True)
        df.rename(columns={0:"counts"}, inplace=True)
        df.drop_duplicates(subset=['seg_id'], inplace=True)

        seg_lengths = self.df.groupby("seg_id").size()
        most_common = np.repeat(df[subtask].to_numpy(), seg_lengths)

        self.df.loc[~self.df["seg_id"].isna(), subtask] = most_common

     
    def step(self, batch):

        if hasattr(self, "pair_data"):
            del self.pair_data

        if hasattr(self, "seg_data"):
            del self.seg_data

        index = np.repeat(range(len(batch)), batch["token"]["lengths"])
        ids = np.repeat(batch.ids, batch["token"]["lengths"])

        to_fill_columns = ["seg_id"] + self.subtasks 

        self.df = pd.DataFrame([], index = index,  columns=to_fill_columns)
        self.stuff = {}
        self.logits = {}
        self.batch = batch

        mask = ensure_numpy(self.batch["token"]["mask"]).astype(bool)

        self.df["sample_id"] = ids
        self.df["text"] = ensure_numpy(batch["token"]["text"])[mask]
        self.df["token_id"] = np.hstack([np.arange(l) for l in ensure_numpy(batch["token"]["lengths"])])
        self.df["original_token_id"] = ensure_numpy(batch["token"]["token_ids"])[mask]
        self.df["T-seg_id"] = ensure_numpy(batch["token"]["seg_id"])[mask]

        for s in self.all_tasks:
            self.df[f"T-{s}"] = ensure_numpy(batch["token"][s])[mask]

        # if we want to use schedule sampling we select the ground truths instead of 
        # the predictions
        self.__use_gt = self.schedule.next(self.batch.current_epoch)

        return self


    def add_stuff(self, stuff):
        self.stuff.update(stuff)


    def add_logits(self, logits:Tensor, task:str):
        self.logits[task] = logits


    def add_preds(self, preds:Union[np.ndarray, Tensor], level:str,  task:str):
        
        mask = ensure_numpy(self.batch[level]["mask"]).astype(bool)

        if level == "token":
            self.df[task] = ensure_numpy(preds[mask])
        
        else:
            seg_preds = ensure_numpy(preds[mask])
            
            # we spread the predictions on segments over all tokens in the segments
            cond = ~self.df["T-seg_id"].isna()
            self.df[task, cond] = np.repeat(seg_preds, ensure_numpy(self.batch["seg"]["lengths"]))


        subtasks = task.split("+")  

        # if our task is complexed, e.g. "seg+label". We decouple the label ids for "seg+label"
        # so we get the labels for Seg and for Label
        if len(subtasks) > 1:

            subtask_preds = self.df[task].apply(lambda x: np.array(self.__decouplers[task][x]))
            subtask_preds = np.stack(subtask_preds.to_numpy())

            for i, subtask in enumerate(subtasks):
                self.df[subtask] = subtask_preds[:,i]
        

        for subtask in subtasks:

            if subtask == "seg":
                self.__decode_segs()
        
            if level == "token":
                self.__ensure_homogeneous(subtask)

            if subtask == "link":
                self.__correct_links(true_segs = level == "seg")    

    @_cache
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
                #print(target)
                j = target["T-seg_id"].to_list()[0]
                seg_ids = target["seg_id"].dropna().to_list()

                if not seg_ids:
                    return np.array([None, None, None])

                i = Counter(seg_ids).most_common(1)[0][0]

                p_index = set(pdf.loc[[i], "token_id"]) #slowest part
                t_index = set(target["token_id"])

                ratio = len(t_index.intersection(p_index)) / max(len(p_index), len(t_index))
                return np.array([i, j, ratio])

            # create pdf with predicted segments ids as index to be able
            # to select rows faster
            pdf = df.copy()
            pdf["index"] = pdf.index 
            pdf.index = pdf["seg_id"]

            # we extract matching information. Which predicted segments are overlapping with which 
            # ground truth segments
            match_info = np.vstack(df.groupby("T-seg_id").apply(overlap, (pdf)))
            
            i = match_info[:,0].astype(int) #predicted segment id
            j = match_info[:,1].astype(int) # ground truth segment id
            ratio = match_info[:,2]

            return i, j, ratio


        key = "seg_id"
        if self.__use_gt:
            key = "T-seg_id"
        
        first_df = self.df.groupby(key, sort=False).first()
        first_df.reset_index(inplace=True)

        last_df = self.df.groupby(key, sort=False).last()
        last_df.reset_index(inplace=True)

        # we create ids for each memeber of the pairs
        p1, p2 = [], []
        j = 0
        for i in range(len(self.batch)):
            n = len(self.df.loc[i,key].dropna().unique())
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
        pair_df["T-link_label"] = first_df.loc[pair_df["p1"], "T-link_label"].to_numpy()

        # find which pairs are "false", i.e. the members whould not be linked
        links = first_df.loc[pair_df["p1"], "T-link"].to_numpy()
        pairs_per_sample = pair_df.groupby("sample_id", sort=False).size().to_numpy()
        seg_per_sample = utils.np_cumsum_zero(first_df.groupby("sample_id", sort=False).size().to_numpy())
        normalized_links  = links + np.repeat(seg_per_sample, pairs_per_sample)
        pair_df["linked"] = first_df.iloc[normalized_links].index.to_numpy() == p2

     
        #set start and end token indexes for p1 and p2
        pair_df["p1_start"] = first_df.loc[pair_df["p1"], "token_id"].to_numpy()
        pair_df["p1_end"] = last_df.loc[pair_df["p1"], "token_id"].to_numpy()

        pair_df["p2_start"] = first_df.loc[pair_df["p2"], "token_id"].to_numpy()
        pair_df["p2_end"] = last_df.loc[pair_df["p2"], "token_id"].to_numpy()

        # set directions
        pair_df["direction"] = 0  #self
        pair_df.loc[pair_df["p1"] < pair_df["p2"], "direction"] = 1 # ->
        pair_df.loc[pair_df["p1"] > pair_df["p2"], "direction"] = 2 # <-

        if not self.__use_gt:
            # we also have information about whether the seg_id is a true segments 
            # and if so, which TRUE segmentent id it overlaps with, and how much
            seg_id, T_seg_id, ratio = extract_match_info(self.df)

            p1_matches = np.isin(pair_df["p1"], seg_id)
            p2_matches = np.isin(pair_df["p2"], seg_id)

            # adding true seg ids for each p1,p2
            i2j = dict(zip(seg_id, T_seg_id))

            p1_v = np.array(p1, dtype=np.float)
            p1_v[~p1_matches] = np.nan

            p2_v = np.array(p2, dtype=np.float)
            p2_v[~p2_matches] =  np.nan

            pair_df["T-p1"] = p1_v
            pair_df["T-p2"] = p2_v
            pair_df["T-p1"] = pair_df["T-p1"].map(i2j)
            pair_df["T-p2"] = pair_df["T-p2"].map(i2j)

            # adding ratio for true seg ids for each p1,p2
            i2ratio = dict(zip(seg_id, ratio))

            p1_ratio_default = np.array(p1, dtype=np.float)
            p1_ratio_default[~p1_matches] = float("-inf")

            p2_ratio_default = np.array(p2, dtype=np.float)
            p2_ratio_default[~p2_matches] = float("-inf")

            pair_df["T-p1-ratio"] = p1_ratio_default
            pair_df["T-p2-ratio"] = p2_ratio_default
            pair_df["T-p1-ratio"] = pair_df["T-p1-ratio"].map(i2ratio)
            pair_df["T-p2-ratio"] = pair_df["T-p2-ratio"].map(i2ratio)
        
        else:
            pair_df["T-p1"] = p1
            pair_df["T-p2"] = p2
            pair_df["T-p1-ratio"] = 1
            pair_df["T-p2-ratio"] = 1

        nodir_pair_df = pair_df[pair_df["direction"].isin([0,1]).to_numpy()]

        pair_dict = {
                    "bidir": {k:torch.tensor(v, device=self.batch.device) for k,v in pair_df.to_dict("list").items()},
                    "nodir": {k:torch.tensor(v, device=self.batch.device) for k,v in nodir_pair_df.to_dict("list").items()}
                    }

        pair_dict["bidir"]["lengths"] = pair_df.groupby("sample_id", sort=False).size().to_list()
        pair_dict["nodir"]["lengths"] = nodir_pair_df.groupby("sample_id", sort=False).size().to_list()

        return pair_dict

    @_cache
    def get_seg_data(self):


        if self.__use_gt:
            seg_data = {
                            "span_idxs": self.batch["seg"]["span_idxs"],
                            "lengths": self.batch["seg"]["lengths"],
                            }   
        else:
            seg_lengths = self.df[~self.df["seg_id"].isna()].groupby(level=0)["seg_id"].nunique().to_numpy()

            start = self.df.groupby("seg_id", sort=False).first()["token_id"]
            end = self.df.groupby("seg_id",  sort=False).last()["token_id"]

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
        
        task_key = task

        # if we want to use schedule sampling we select the ground truths instead of 
        # the predictions
        if self.__use_gt:
            task_key = f"T-{task}"

        # we take the lengths in tokens for each sample
        tok_lengths = self.df.groupby(level=0).size().to_numpy()

        # create the end indexes. Remove the last value as its the end and will create a faulty split
        end_idxs = np.cumsum(tok_lengths)[:-1]

        # split and then pad
        preds = utils.zero_pad(np.hsplit(self.df[task_key].to_numpy(), end_idxs))

        #then we create one_hots from the preds if thats needed
        if one_hot:
            return utils.one_hot(
                                    matrix = torch.LongTensor(preds),
                                    mask = self.batch["token"]["mask"],
                                    num_classes = len(self.label_encoders[task])
                                    )
        else:
            return preds
        







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