

#basics
from typing import Union, List
import numpy as np
from numpy.lib import utils
import pandas as pd
from collections import Counter


#segnlp
from .input import Input
from .array import ensure_flat, ensure_numpy, flatten
from .bio_decoder import BIODecoder
from .schedule_sample import ScheduleSampling

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

            for label in labels:

                sublabels = label.split("_")
                decouplers[task][label] = [label_encoders[st].decode(sl) for st, sl in zip(subtasks, sublabels)]

        return decouplers
    

    def __set(self, i, key, value):
        self.df.loc[i, key] = value

    
    def __unfold_over_seg(self, i, key, data, gt:bool=False):

        sample = self.df.loc[i]
        sample.index = np.arange(sample.shape[0])

        groups = sample.groupby("T-seg_id" if gt else "seg_id")

        seg_id =  [i for i,g in groups][0]

        idx = np.hstack(list(groups.groups.values()))
        lengths = groups.size().to_numpy()

        values = np.repeat(data, lengths)

        sample.loc[idx, key] = values

        self.df.loc[i, key] = sample[key].to_numpy()


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


    def __decouple(self, subtask_preds:np.ndarray, task:str):
        decouple = lambda x: self.__decouplers[task][x] #.split("_") #[subtask_position]
        return zip(*[decouple(x) for x in subtask_preds])


    def __correct_links(self, i, max_seg:int):
        """
        This function perform correction 3 mentioned in https://arxiv.org/pdf/1704.06104.pdf  (Appendix)

        Any link that is outside of the actuall text, e.g. when predicted link > max_idx, is set to predicted_link== max_idx
        """
    
        links_above_allowed = self.df.loc[i, "link"] > max_seg
        self.df.loc[i,"link"][links_above_allowed] = max_seg

        links_bellow_allowed = self.df.loc[i, "link"] < 0
        self.df.loc[i,"link"][links_bellow_allowed] = max_seg


    def __ensure_homogeneous(self, i, subtask):

        """
        ensures that the labels inside a segments are the same. For each segment we take the majority label 
        and use it for the whole span.
        """
        self.df.loc[i,subtask] = self.df.loc[i].groupby("seg_id")[subtask].transform(lambda x:x.value_counts().index[0])

     
    def step(self, batch):
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
        self.df["token_id"] = ensure_numpy(batch["token"]["token_ids"])[mask]
        self.df["T-seg_id"] = ensure_numpy(batch["token"]["seg_id"])[mask]

        for s in self.subtasks:
            self.df[f"T-{s}"] = ensure_numpy(batch["token"][s])[mask]



            # if not hasattr(self, "_first_done"):
            #     self.__set(
            #                 i, 
            #                 "sample_id", 
            #                 self.batch.ids[i]
            #                 )

            #     self.__set(
            #                 i, 
            #                 "text", 
            #                 self.batch["token"]["text"][i, :tok_lengths[i]]
            #                 )

            #     self.__set(
            #                 i, 
            #                 "token_id", 
            #                 self.batch["token"]["token_ids"][i, :tok_lengths[i]]
            #             )
                
            #     self.__set(
            #                 i,
            #                 "T-seg_id", 
            #                 self.batch["token"]["seg_id"][i, :tok_lengths[i]]
            #                 )



        return self


    def add_stuff(self, stuff):
        self.stuff.update(stuff)


    def add_logits(self, logits:Tensor, task:str):
        self.logits[task] = logits


    def add_preds(self, preds:Union[np.ndarray, Tensor], level:str,  task:str):
        
        preds = ensure_numpy(preds)
        global_seg_id = 0
        for i in range(len(preds)):

            sample_preds = preds[i][:self.batch[level]["lengths"][i]]

            # if task is complexed, i.e. combination of two or more subtasks, we decouple the labels for each 
            # subtask
            if "+" in task:
                decoupled_preds = self.__decouple(
                                                sample_preds = sample_preds,
                                                task = task 
                                                )
                pred_dict =  dict(zip(task.split("+"), decoupled_preds))
            else:
                pred_dict = {task: sample_preds}
            
            if "seg" in pred_dict:
                tok_seg_ids, _ = self.seg_decoder(
                                                pred_dict.pop("seg"), 
                                                seg_id_start = global_seg_id
                                                )
                self.__set(i, "seg_id", tok_seg_ids)

                max_seg = len(self.df.loc[i].groupby("seg_id"))
            else:
                max_seg = len(self.df.loc[i].groupby("T-seg_id"))


            for subtask, subtask_preds in pred_dict.items():
                
                # we are unfolding the segment preditions to tokens. This is only so we can fit the predictions into a standardised token-based df
                if level == "seg":
                    self.__unfold_over_seg(i, subtask, subtask_preds, gt=True)
                else:
                    self.__set(i, subtask, subtask_preds)

                    self.__ensure_homogeneous(i)


            if subtask == "link":
                self.__correct_links(
                                    i,
                                    max_seg = max_seg
                                    )



                # if we are predicting on token level we make select the majority label for each segment
                # #if level == "token":
                #     # sample_preds = self.__ensure_homogeneous(
                #     #                                         subtask_preds = subtask_preds,
                #     #                                         span_token_lengths = span_token_lengths,
                #     #                                         none_span_mask = none_span_mask
                #     #                                         )

                # # we dont decode the links, and if we are allowign prediction of links outside the scope of the segments, example when encoding links into complexed labels.
                # # we want to correct this. ALl links which point to outside the predicted amount of segments will be set to point to the last segment in the sample
                # # if subtask == "link":
                # #     sample_preds = self.__correct_links(
                # #                                         subtask_preds = subtask_preds,
                # #                                         max_seg = len(self.df.loc[i].groupby("seg_id"))
                # #                                         )


                # # we are unfolding the segment preditions to tokens. This is only so we can fit the predictions into a standardised token-based df
                # if level == "seg":
                #     self.__unfold_over_seg(i, task, sample_preds, gt=True)
                # else:
                #     self.__set(i, subtask, sample_preds)

     
    def get_pairs(self, bidir:bool = False):
        pass


def get_all_possible_pairs(
                            start: List[List[int]],
                            end: List[List[int]],
                            device=torch.device,
                            bidir=False,
                        ) -> DefaultDict[str, List[List[Tuple[int]]]]:

    if bidir:
        get_pairs = lambda x, r: list(product(x, repeat=r))  # noqa
    else:
        get_pairs = lambda x, r: sorted(  # noqa
            list(combinations(x, r=r)) + [(e, e) for e in x],
            key=lambda u: (u[0], u[1]))

    all_possible_pairs = defaultdict(lambda: [])
    for idx_start, idx_end in zip(start, end):
        idxs = list(get_pairs(range(len(idx_start)), 2))
        if idxs:
            p1, p2 = zip(*idxs)  # pairs' seg id
            p1 = torch.tensor(p1, dtype=torch.long, device=device)
            p2 = torch.tensor(p2, dtype=torch.long, device=device)
            # pairs start and end token id.
            start = get_pairs(idx_start, 2)  # type: List[Tuple[int]]
            end = get_pairs(idx_end, 2)  # type: List[Tuple[int]]
            lens = len(start)  # number of pairs' segs len  # type: int

        else:
            p1 = torch.empty(0, dtype=torch.long, device=device)
            p2 = torch.empty(0, dtype=torch.long, device=device)
            start = []
            end = []
            lens = 0

        all_possible_pairs["idx"].append(idxs)
        all_possible_pairs["p1"].append(p1)
        all_possible_pairs["p2"].append(p2)
        all_possible_pairs["start"].append(start)
        all_possible_pairs["end"].append(end)
        all_possible_pairs["lengths"].append(lens)

    all_possible_pairs["lengths"] = torch.tensor(all_possible_pairs["lengths"],
                                                 dtype=torch.long,
                                                 device=device)
    all_possible_pairs["total_pairs"] = sum(all_possible_pairs["lengths"])
    return all_possible_pairs


    # def get_stuff(self, key:str):
    #     return self.stuff[key]


    # def get_logits(self, task:str):
    #     return self.logits[task]


    # def get_preds(self, task:str):

    #     seg_id = "seg_id"

    #     if self.schedule.next(self.batch.current_epoch):
    #         task = f"T-{task}"
    #         seg_id = f"T-seg_id"
        
    #     preds_splits = np.split(self.df[task].to_numpy(), self.df.groupby("seg_id").size().to_numpy())
    #     preds = utils.zero_pad(preds_splits)
    #     return preds
        







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