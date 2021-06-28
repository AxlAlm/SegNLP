

#basics
from inspect import getsource
from re import sub
from typing import Union, List
from unittest import result
import numpy as np
from numpy.lib import utils
import pandas as pd
from collections import Counter
from itertools import product
from itertools import combinations

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
                segment_sampling:bool=False,
                sampling_k:int=5,
                ):

        self.inference = inference
        #self.label_encoders = label_encoders
        self.tasks = tasks
        self.all_tasks = all_tasks
        self.prediction_level = prediction_level
        self.subtasks = subtasks

        
        self.schedule = segment_sampling
        if segment_sampling:
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
    

    def __extract_match_info(self):
        

        def overlap(target, pdf):
            j = target["T-seg_id"].to_list()[0]
            seg_ids = target["seg_id"].dropna().to_list()

            if not seg_ids:
                return [0, 0, -1, -1]

            i = Counter(seg_ids).most_common(1)[0][0]

            p_index = set(pdf.loc[[i], "index"]) #slowest part
            t_index = set(target.index)

            ratio = len(t_index.intersection(p_index)) / max(len(p_index), len(t_index))
            return ratio


        # create pdf with predicted segments ids as index to be able
        # to select rows faster
        pdf = self.df.copy()
        pdf["index"] = pdf.index 
        pdf.index = pdf["seg_id"]

        # we extract matching information. Which predicted segments are overlapping with which 
        # ground truth segments
        match_info = np.vstack(self.df.groupby("T-seg_id").apply(overlap, (pdf)))
        ratio = match_info[:,0].astype(bool)
        i = match_info[:,1] #predicted segment id
        j = match_info[:,2] # ground truth segment id
        return i, j, ratio


    def __decode_segs(self):
    
        # we get the sample start indexes from sample lengths. We need this to tell de decoder where samples start
        sample_sizes = self.df.groupby("sample_id").size().to_numpy()
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

        max_segs = self.df.groupby("sample_id")["T-seg_id" if true_segs else "seg_id"].nunique().to_numpy()
        self.df["max_seg"] = np.repeat(max_segs, df.groupby("sample_id").size().to_numpy())

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

        for s in self.subtasks:
            self.df[f"T-{s}"] = ensure_numpy(batch["token"][s])[mask]

        return self


    def add_stuff(self, stuff):
        self.stuff.update(stuff)


    def add_logits(self, logits:Tensor, task:str):
        self.logits[task] = logits

    @utils.timer
    def add_preds(self, preds:Union[np.ndarray, Tensor], level:str,  task:str):
        
        mask = ensure_numpy(self.batch[level]["mask"]).astype(bool)

        if level == "token":
            self.df[task] = ensure_numpy(preds[mask])
        
        else:
            seg_preds = ensure_numpy(preds[mask])
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



     
    def get_pair_data(self, bidir:bool = False):
        

        def create_pair_data(
                                df:pd.DataFrame,
                                bidir = False,
                                device = torch.device
                            ) -> DefaultDict[str, List[List[Tuple[int]]]]:

            if bidir:
                get_pairs = lambda x, r: list(product(x, repeat=r))  # noqa
            else:
                get_pairs = lambda x, r: sorted(  # noqa
                                                list(combinations(x, r=r)) + [(e, e) for e in x],
                                                key=lambda u: (u[0], u[1])
                                                )

            start_token_idxs = ""
            end_token_idxs = ""


            pair_data = defaultdict(lambda: [])
            for idx_start, idx_end in zip(start_token_idxs, end_token_idxs):
                
                # create indexes for pairs 
                idxs = list(get_pairs(range(len(idx_start)), 2))

                if idxs:
                    p1, p2 = zip(*idxs)  # pairs' seg id
                    p1 = torch.tensor(p1, dtype=torch.long, device=device)
                    p2 = torch.tensor(p2, dtype=torch.long, device=device)
                    # pairs start and end token id.
                    start = get_pairs(idx_start, 2)  # type: List[Tuple[int, int]]
                    end = get_pairs(idx_end, 2)  # type: List[Tuple[int, int]]
                    lens = len(start)  # number of pairs' segs len  # type: int

                else:
                    p1 = torch.empty(0, dtype=torch.long, device=device)
                    p2 = torch.empty(0, dtype=torch.long, device=device)
                    start = []
                    end = []
                    lens = 0

                pair_data["idx"].append(idxs)
                pair_data["p1"].append(p1)
                pair_data["p2"].append(p2)
                pair_data["start"].append(start)
                pair_data["end"].append(end)
                pair_data["lengths"].append(lens)

            pair_data["lengths"] = torch.tensor(
                                                pair_data["lengths"],
                                                dtype=torch.long,
                                                device=device
                                                )
            pair_data["total_pairs"] = sum(pair_data["lengths"])


            return pair_data


        #simple cacheing
        if not hasattr(self, "pair_data"):
            self.pair_data = create_pair_data(
                                            df = self.df, 
                                            bidir = bidir,
                                            device = self.batch.device
                                            )
        

        elif self.pair_data["bidir"] != bidir:
             self.pair_data = create_pair_data(
                                            df = self.df, 
                                            bidir = bidir,
                                            device = self.batch.device
                                            )


        return self.pair_data


    @utils.timer
    def get_seg_data(self):
        
        #simple cacheing
        if not hasattr(self, "seg_data"):

            
            # X = []
            # prev_seg_id = -1
            # for i,row in self.df.iterrows():
                
            #     if prev_seg_id != row["seg_id"]:
            #         prev_seg_id = row["seg_id"]
            #         X.append(row["seg_id"])

            
            # print(Counter(X).most_common(10))

                
            print(len(set(self.df["seg_id"])))
        
            x = np.sum(self.df.groupby("sample_id")["T-seg_id"].nunique(dropna=True).to_numpy())
            y = len(self.df.groupby("T-seg_id").first()["token_id"])
            assert x == y

            print(set(self.df[~self.df["seg_id"].isna()]["seg_id"]))
            seg_lengths = self.df[~self.df["seg_id"].isna()].groupby("sample_id")["seg_id"].nunique().to_numpy()

            print("SEG LENGTHS", np.sum(seg_lengths), np.sum(self.df.groupby("sample_id")["seg_id"].nunique().to_numpy()))

            start = self.df.groupby("seg_id").first()["token_id"]
            end = self.df.groupby("seg_id").last()["token_id"]

            print("START", len(start))
            print("END", len(end))

            span_idxs_flat = list(zip(start, end))

            assert len(span_idxs_flat) == np.sum(seg_lengths), f"{len(span_idxs_flat)} {np.sum(seg_lengths)}"

            span_idxs = np.zeros((len(self.batch), np.max(seg_lengths), 2))
            floor = 0
            for i,l in enumerate(seg_lengths):

                print(i,l, floor)
                span_idxs[i][:l] = np.array(span_idxs_flat[floor:floor+l])
                floor += l

            print(span_idxs)



            self.seg_data = {
                            "span_idxs": span_idxs,
                            "lengths": nr_segs
                            }   


        return self.seg_data
        

    # def get_stuff(self, key:str):
    #     return self.stuff[key]


    # def get_logits(self, task:str):
    #     return self.logits[task]


    def get_preds(self, task:str, one_hot:bool = False):

        seg_id = "seg_id"

        if self.schedule.next(self.batch.current_epoch):
            task = f"T-{task}"
            seg_id = f"T-seg_id"

        
        
        nr_segs = self.df.groupby("sample_id")["seg_id"].nunique()

        
        preds_splits = np.split(self.df[task].to_numpy(), self.df.groupby("seg_id").size().to_numpy())
        preds = utils.zero_pad(preds_splits)
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