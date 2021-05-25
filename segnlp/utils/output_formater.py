

#basics
from typing import Union, List
import numpy as np
import pandas as pd
from collections import Counter


#segnlp
from .model_input import ModelInput
from .array import ensure_flat, ensure_numpy, flatten
from .bio_decoder import BIODecoder

#pytorch
import torch


class OutputFormater:

    def __init__(self, 
                label_encoders:dict,
                tasks:list,
                all_tasks:list,
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



    def __get_token_seg_ids(self,
                            span_token_lengths:np.ndarray, 
                            none_span_masks:np.ndarray,
                            seg_lengths:np.ndarray,
                            span_lengths:np.ndarray
                            ) -> list:

        batch_token_seg_ids = []
        bz = len(span_token_lengths)
        total_segs = 0
        for i in range(bz):
            nr_seg = seg_lengths[i]
            span_tok_lens = ensure_numpy(span_token_lengths[i][:span_lengths[i]])
            seg_mask = ensure_numpy(none_span_masks[i][:span_lengths[i]])
            seg_ids = np.arange(start=total_segs, stop=total_segs+nr_seg)
            seg_mask[seg_mask.astype(bool)] += seg_ids

            ids = ensure_numpy(seg_mask).astype(float)
            ids[ids == 0] = float("NaN")
            ids[ids != float("NaN")] -= 1

            token_seg_ids = np.repeat(ids, span_tok_lens)
            batch_token_seg_ids.extend(token_seg_ids.tolist())
            total_segs += nr_seg

        return batch_token_seg_ids


    def __add_subtask_preds(self, decoded_labels:np.ndarray, lengths:np.ndarray, level:str,  task:str):
        """
        given that some labels are complexed, i.e. 2 plus tasks in one, we can break these apart and 
        get the predictions for each of the task so we can get metric for each of them. 

        for example, if one task is Segmentation+Arugment Component Classification, this task is actually two
        tasks hence we can break the predictions so we cant get scores for each of the tasks.
        """


        #sum_lens = np.sum(lengths)
        max_len = max(lengths)
        size = decoded_labels.shape[0]
        subtasks_predictions = {}
        subtasks = task.split("+")
        for subtask in subtasks:
            #print("__________________________________________")
            #print(subtask)

            subtask_position = subtasks.index(subtask)

            if subtask == "link":
                dtype = np.int16
                decoder = lambda x: int(str(x).split("_")[subtask_position])
            else:
                dtype = decoded_labels.dtype
                decoder = lambda x: str(x).split("_")[subtask_position]


            subtask_preds = np.zeros((size, max_len), dtype=dtype)
            for i in range(size):
                
                #print(decoded_labels[i][:lengths[i]])
                #print([decoder(x) for x in decoded_labels[i][:lengths[i]]])
                subtask_preds[i][:lengths[i]] = [decoder(x) for x in decoded_labels[i][:lengths[i]]]


            self.add_preds(
                            task=subtask, 
                            level=level, 
                            data=subtask_preds,
                            decoded=True,
                            )
    

    def __decode_labels(self, preds:np.ndarray, lengths:np.ndarray, task:str):

        size = preds.shape[0]
        decoded_preds = np.zeros((size, max(lengths)), dtype="<U30")
        for i in range(size):
            decoded_preds[i][:lengths[i]] = self.label_encoders[task].decode_list(preds[i][:lengths[i]])
        
        return decoded_preds


    def __decode_token_link_labels(self, preds:np.ndarray, lengths:np.ndarray, span_lengths:np.ndarray, none_spans:np.ndarray):
        
        size = preds.shape[0]
        decoded_preds = np.zeros((size, max(lengths)), dtype=np.int16)
        for i in range(size):
            decoded = self.label_encoders["link"].decode_token_links(
                                                                    preds[i][:lengths[i]], 
                                                                    span_token_lengths=span_lengths[i],
                                                                    none_spans=none_spans[i]
                                                                    )
            decoded_preds[i][:lengths[i]] = decoded
        
        return decoded_preds


    def __handle_complex_tasks(self, data:torch.tensor, level:str, lengths:np.ndarray, task:str):

        # if task is a complex task we get the predictions for each of the subtasks embedded in the 
        # complex task. We will only use these to evaluate
        decoded_labels = self.__decode_labels(
                                                preds=data, 
                                                lengths=lengths,
                                                task=task,
                                                )
        

        subtask_preds = self.__add_subtask_preds(
                                                decoded_labels=decoded_labels,
                                                lengths=lengths,
                                                level=level,
                                                task=task
                                                )
        
        # for stask, sdata in subtask_preds.items():
        #     self.add_preds(
        #                     task=stask, 
        #                     level=level, 
        #                     data=sdata,
        #                     decoded=True,
        #                     )


    def __unfold_span_labels(self, span_labels:np.ndarray, span_indexes:np.ndarray, max_nr_token:int, fill_value:int):
        
        batch_size = span_labels.shape[0]
        tok_labels = np.full((batch_size, max_nr_token), fill_value=fill_value)
        for i in range(batch_size):
            for j,(start,end) in enumerate(span_indexes[i]):
                tok_labels[i][start:end+1] = span_labels[i][j]
        return tok_labels
        
    
    def __correct_token_links(self, 
                                        links:np.ndarray, 
                                        lengths_segs:np.ndarray,
                                        span_token_lengths:np.ndarray, 
                                        none_spans:np.ndarray, 
                                        decoded:bool,
                                        ):
        """
        This function perform correction 3 mentioned in https://arxiv.org/pdf/1704.06104.pdf  (Appendix)

        Any link that is outside of the actuall text, e.g. when predicted link > max_idx, is set to predicted_link== max_idx

        """

        
        new_links = []
        for i in range(links.shape[0]):
            last_seg_idx = max(0, lengths_segs[i]-1)
            sample_links = links[i]

            if decoded:

                sample_links = self.label_encoders["link"].encode_token_links(
                                                                                sample_links,
                                                                                span_token_lengths=span_token_lengths[i],
                                                                                none_spans=none_spans[i]
                                                                                )
       
            links_above_allowed = sample_links > last_seg_idx
            sample_links[links_above_allowed] = last_seg_idx

            links_above_allowed = sample_links < 0
            sample_links[links_above_allowed] = last_seg_idx

            # sample_links = self.label_encoders["link"].decode_token_links(
            #                                                                 sample_links,
            #                                                                 span_token_lengths=span_token_lengths[i],
            #                                                                 none_spans=none_spans[i]
            #                                                                 )

            new_links.append(sample_links)
        
        return np.array(new_links)


    def __make_homogeneous(self,
                            data:np.ndarray,
                            span_token_lengths:np.ndarray,
                            none_spans:np.ndarray,
                            ):

        for i in range(data.shape[0]):
            
            s = 0
            for j,l in enumerate(span_token_lengths[i]):

                if none_spans[i][j] == 0:
                    pass
                else:
                    majority_label = Counter(data[i][s:s+l]).most_common(1)[0][0]
                    data[i][s:s+l] = majority_label
                s += l
        
        return data


    def __add_preds(self, task:str, level:str, data:torch.tensor, decoded:bool=False):

        #assert task in set(self.tasks), f"{task} is not a supported task. Supported tasks are: {self.tasks}"
        assert level in set(["token", "seg"]), f"{level} is not a supported level. Only 'token' or 'seg' are supported levels"
        assert torch.is_tensor(data) or isinstance(data, np.ndarray), f"{task} preds need to be a tensor or numpy.ndarray"
        assert len(data.shape) == 2, f"{task} preds need to be a 2D tensor"

        data = ensure_numpy(data)

        if "+" in task:
            self.__handle_complex_tasks(
                                        data=data,
                                        level=level,
                                        lengths=ensure_numpy(self.batch[level]["lengths"]),
                                        task=task
                                        )
            return

        if level == "token" and task != "seg":
            # correct all the labels within a segments so that they are all the same using majority rule
            # e.g. if the labels within a segment differ we take the majoriy label and use it as 
            # a label for all tokens in the segment
            data = self.__make_homogeneous(
                                            data,
                                            span_token_lengths=ensure_numpy(self.pred_seg_info["span"]["lengths_tok"]),
                                            none_spans=ensure_numpy(self.pred_seg_info["span"]["none_span_mask"]),
                                            )


        if level == "seg":

            span_indexes = ensure_numpy(self.batch["seg"]["span_idxs"])
            data = self.__unfold_span_labels(
                                            span_labels=data,
                                            span_indexes=span_indexes,
                                            max_nr_token=max(ensure_numpy(self.batch["token"]["lengths"])),
                                            fill_value= 0 if task == "link" else -1
                                            )

            if not self._pred_span_set:
                self.pred_seg_info["seg"]["lengths"] = self.batch["seg"]["lengths"]
                self.pred_seg_info["span"]["lengths"] = self.batch["span"]["lengths"]
                self.pred_seg_info["span"]["lengths_tok"] = self.batch["span"]["lengths_tok"]
                self.pred_seg_info["span"]["none_span_mask"] = self.batch["span"]["none_span_mask"]
                self._pred_span_set = True

            level = "token"


        if task == "link":
            preds = self.__correct_token_links(
                                                        data,
                                                        lengths_segs=ensure_numpy(self.pred_seg_info["seg"]["lengths"]),
                                                        span_token_lengths=ensure_numpy(self.pred_seg_info["span"]["lengths_tok"]),
                                                        none_spans=ensure_numpy(self.pred_seg_info["span"]["none_span_mask"]),
                                                        decoded=decoded,
                                                        )
        else:
            if decoded:
                preds = data
            else:
                preds = self.__decode_labels(
                                                    preds=data, 
                                                    lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                    task=task
                                                    )

        if not self.inference:

            if task == "link":
                targets = ensure_numpy(self.batch[level][task])
            else:
                targets = self.__decode_labels(
                                                preds=ensure_numpy(self.batch[level][task]), 
                                                lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                task=task
                                                )


        if task == "seg":
            seg_data = self.seg_decoder(
                                        batch_encoded_bios=preds,
                                        lengths=ensure_numpy(self.batch[level]["lengths"]),
                                        )
            self.pred_seg_info["seg"]["lengths"] = seg_data["seg"]["lengths"]
            self.pred_seg_info["span"]["lengths"] = seg_data["span"]["lengths"]
            self.pred_seg_info["span"]["lengths_tok"] = seg_data["span"]["lengths_tok"]
            self.pred_seg_info["span"]["none_span_mask"] = seg_data["span"]["none_span_mask"]
            

            token_seg_ids = self.__get_token_seg_ids(
                                                        span_token_lengths = self.pred_seg_info["span"]["lengths_tok"],
                                                        none_span_masks = self.pred_seg_info["span"]["none_span_mask"],
                                                        seg_lengths = self.pred_seg_info["seg"]["lengths"],
                                                        span_lengths = self.pred_seg_info["span"]["lengths"],
                                                        )
            self._outputs["seg_id"] = token_seg_ids

        mask = ensure_numpy(self.batch[level]["mask"])
        preds_flat = ensure_flat(ensure_numpy(preds), mask=mask)
        targets_flat = ensure_flat(ensure_numpy(targets), mask=mask)

        self._outputs[task] = pred_outputs
        self._outputs[f"T-{task}"] = target_outputs


    def __init__output(self, input):

        # INIT
        self.batch = input
        self._outputs = {}
        self.pred_seg_info = {
                                "seg":{},
                                "span":{},
                                }
        self._pred_span_set = False


        self._outputs["sample_id"] = np.concatenate([np.full(int(l),int(i)) for l,i in zip(self.batch["token"]["lengths"],self.batch.ids)])
        self._outputs["text"] = ensure_flat(ensure_numpy(self.batch["token"]["text"]), mask=ensure_numpy(self.batch["token"]["mask"])),

        token_ids = [self.batch["token"]["token_ids"][i,:self.batch["token"]["lengths"][i]] for i in range(len(batch))]
        self._outputs["token_id"] = ensure_numpy([int(t) for sub in token_ids for t in sub])

        if not self.inference:  
            self._outputs["T-seg_id"] = self.__get_token_seg_ids(
                                                                span_token_lengths = self.batch["span"]["lengths_tok"],
                                                                none_span_masks = self.batch["span"]["none_span_mask"],
                                                                seg_lengths = self.batch["seg"]["lengths"],
                                                                span_lengths = self.batch["span"]["lengths"],
                                                            )


    def format(self, input, output):

        self.__init__output(input)

        for task, level, data, type_  in output.items():

            if type_ == "pred":
                self.__add_preds(
                                task=task, 
                                level=level,
                                data=data,
                                )
            else:
                raise NotImplementedError(f'"{type_}" is not suported')

        return self._outputs