

#basics
from typing import Union, List
import numpy as np
import pandas as pd
from collections import Counter


#segnlp
from .model_input import ModelInput
from .array import ensure_flat, ensure_numpy, flatten
from .bio_decoder import BIODecoder
from segnlp import utils

#pytorch
import torch
from torch import Tensor



class Output:

    def __init__(self, batch, subtasks):
        index = np.repeat(range(len(batch)), batch["token"]["lengths"])
        columns = ["sample_id", "text", "token_id"] + subtasks + [f"T-{t}" for t in subtasks] + ["seg_id", "T-seg_id"]
        self.df  = pd.DataFrame([], index = index,  columns=columns)


    def set(self, i, key, value):
        self.df.loc[i, key] = value

    
    def unfold_over_seg(self, i, key, data, gt:bool=False):
        groups = self.df.groupby("T-seg_id" if gt else "seg_id")
        idx = np.hstack(groups.groups.values())
        values = np.repeat(data, groups.size().to_numpy())
        self.df.loc[idx, key] = values        


    def extract_match_info(self):
        

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
        
        # #contains mapping between i j where i is an exact/approx match for j
        # i2j_exact = dict(zip(i[exact],j[exact]))
        # i2j_approx = dict(zip(i[approx],j[approx]))


        return exact, approx, i2j_exact, i2j_approx, i


    def pairs(self, bidir:bool = False):
        pass





class OutputFormater:

    def __init__(self, 
                label_encoders:dict,
                tasks:list,
                all_tasks:list,
                subtasks:list,
                prediction_level:str,
                inference:bool,
                ):

        self.inference = inference
        self.label_encoders = label_encoders
        self.tasks = tasks
        self.all_tasks = all_tasks
        self.prediction_level = prediction_level

        
        seg_task = [task for task in self.tasks if "seg" in task]
        if seg_task:
            id2label = label_encoders[seg_task[0]].id2label
            self.seg_decoder = BIODecoder(
                                        B = [i for i,l in id2label.items() if "B-" in l],
                                        I = [i for i,l in id2label.items() if "I-" in l],
                                        O = [i for i,l in id2label.items() if "O-" in l],
                                        )


    def __decode(self, sample_data, subtask_position, task):
        decoder = lambda x: self.label_encoders[task].decode(x).split("_")[subtask_position]
        return [decoder(x) for x in sample_data]


    def __unfold(self, sample_data:np.ndarray, span_lengths:np.ndarray, none_span_mask:np.ndarray, fill_value:int):
        span_labels = np.full(none_span_mask.shape, fill_value=fill_value)
        span_labels[none_span_mask] = sample_data
        token_labels = np.repeat(span_labels, span_lengths)
        return token_labels


    def __correct_links(self, sample_data, max_seg):

        """
        This function perform correction 3 mentioned in https://arxiv.org/pdf/1704.06104.pdf  (Appendix)

        Any link that is outside of the actuall text, e.g. when predicted link > max_idx, is set to predicted_link== max_idx

        """
        #last_seg_idx = max(0, lengths_segs[i]-1)
    
        links_above_allowed = sample_data > max_seg
        sample_data[links_above_allowed] = max_seg

        links_above_allowed = sample_data < 0
        sample_data[links_above_allowed] = max_seg

        return sample_data


    def __ensure_homogeneous(self, sample_data, span_token_lengths, none_span_mask):

        """
        ensures that the labels inside a segments are the same. For each segment we take the majority label 
        and use it for the whole span.
        """
  
        s = 0
        for j,l in enumerate(span_token_lengths):

            if none_span_mask[j] == 0:
                pass
            else:
                majority_label = Counter(sample_data[s:s+l]).most_common(1)[0][0]
                sample_data[s:s+l] = majority_label
            s += l
    
        return sample_data
     

 

    def fill(self, output:Output, batch:Input, data:Union[np.ndarray, Tensor], level:str,  task:str, first=False):
        """
        given that some labels are complexed, i.e. 2 plus tasks in one, we can break these apart and 
        get the predictions for each of the task so we can get metric for each of them. 

        for example, if one task is Segmentation+Arugment Component Classification, this task is actually two
        tasks hence we can break the predictions so we cant get scores for each of the tasks.
        """

        data = ensure_numpy(data)
        lengths = batch[level]["length"]
        global_seg_id = 0
        for i in range(len(data)):

            sample_data = data[i][:lengths[i]]

            if not hasattr(output, "_first_done"):
                output.set(i, "sample_id", self.batch.ids[i])
                output.set(i, "text", self.batch["token"]["text"][i, :lengths[i]])
                output.set(i, "token_id", self.batch["token"]["token_ids"][i, :lengths[i]])
                output.set(i, "T-seg_id", np.repeat(range(global_seg_id, self.batch["seg"]["length"][i])), self.batch["seg"]["tok_lengths"][i]))
                output._first_done = True

            for k, subtask in enumerate(task.split("+")):
                
                # if we are predicting on token level we make select the majority label for each segment
                if level == "token":
                    sample_data = self.__ensure_homogeneous(sample_data)
        
                # we dont decode the links, and if we are allowign prediction of links outside the scope of the segments, example when encoding links into complexed labels.
                # we want to correct this. ALl links which point to outside the predicted amount of segments will be set to point to the last segment in the sample
                if task != "link":
                    sample_data = self.__correct_links(
                                                        sample_data = sample_data,
                                                        max_seg = len(output.df.loc[i].groupby("seg_id"))
                                                        )
                else:
                    sample_data = self.__decode(
                                                sample_data = sample_data, 
                                                subtask_position = k
                                                )


                # we are unfolding the segment preditions to tokens. This is only so we can fit the predictions into a standardised token-based df
                if level == "seg":
                    output.unfold_over_seg(i, sample_data, task, gt=True)

                # we decode segments
                if task == "seg":
                    tok_seg_ids, _ = self.seg_decoder(sample_data, seg_id_start = global_seg_id)
                    output.set(i, "seg_id", tok_seg_ids)
                    output.unfold_over_seg(i, sample_data, task , gt = False)
                    

                output.set(i, subtask, sample_data)

           











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

        self.__init__output(batch)

        for task, data in preds.items():

            if data.shape[-1] == self.max_seg:
                level = "seg"
            elif data.shape[-1] == self.max_tok:
                level = "token"
            else:
                raise RuntimeError(f"data was given in shape {data.shape[0]} but {self.max_seg} (segment level) or {self.max_tok} (token level) was expected")

            self.__add_preds(
                            task=task, 
                            level=level,
                            data=data,
                            )
         
        return pd.DataFrame(self._outputs)