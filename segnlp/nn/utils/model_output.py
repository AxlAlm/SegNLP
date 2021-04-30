

#basics
from typing import Union, List
import numpy as np
import pandas as pd


#segnlp
from segnlp.metrics import token_metrics
from . import ModelInput
from segnlp.utils import ensure_flat, ensure_numpy
from segnlp.nn.utils import bio_decode

# from segnlp.visuals.tree_graph import arrays_to_tree
# from segnlp.visuals.text_markers import highlight_text

#pytorch
import torch




class ModelOutput:

    def __init__(self, 
                batch:ModelInput,
                return_output:Union[List[int], bool], 
                label_encoders:dict,
                tasks:list,
                all_tasks:list,
                prediction_level:str,
                inference:bool,
                ):

        self.return_output = return_output
        self.inference = inference
        self.label_encoders = label_encoders
        self.tasks = tasks
        self.all_tasks = all_tasks
        self.prediction_level = prediction_level
        self.batch = batch
        self._total_loss_added = False

        self.loss = {}
        self.metrics = {}
        self.outputs = {
                        "sample_idx":np.concatenate([np.full(int(l),int(i)) for l,i in zip(self.batch["token"]["lengths"],self.batch.ids)]),
                        "text": ensure_flat(ensure_numpy(self.batch["token"]["text"]), mask=ensure_numpy(self.batch["token"]["mask"]))
                        }
        self.pred_spans = {}
        self._pred_span_set = False
        self.mask = []
        

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

            subtask_position = subtasks.index(subtask)

            if subtask == "link":
                dtype = np.int16
                decoder = lambda x: int(str(x).split("_")[subtask_position])
            else:
                dtype = decoded_labels.dtype
                decoder = lambda x: str(x).split("_")[subtask_position]


            subtask_preds = np.zeros((size, max_len), dtype=dtype)
            for i in range(size):
                subtask_preds[i][:lengths[i]] = [decoder(x) for x in decoded_labels[i][:lengths[i]] ]


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
            # print(  
            #         preds[i].shape,
            #         lengths[i],
            #         preds[i][:lengths[i]].shape,
            #         lengths[i],
            #         len(self.label_encoders[task].decode_list(preds[i][:lengths[i]])), 
            #         lengths[i],
            #         decoded_preds[i][:lengths[i]].shape
            #     )
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


    def __unfold_span_labels(self, span_labels:np.ndarray, span_indexes:np.ndarray, max_nr_token:int):
        
        batch_size = span_labels.shape[0]
        tok_labels = np.zeros((batch_size, max_nr_token))
        for i in range(batch_size):
            for j,(start,end) in enumerate(span_indexes[i]):
                tok_labels[i][start:end] = span_labels[i][j]
        return tok_labels
        
    
    def __correct_links(self,   links:np.ndarray, 
                                lengths_units:np.ndarray,
                                span_token_lengths:np.ndarray, 
                                none_spans:np.ndarray, 
                                decoded:bool):
        """
        Any link that is outside of the actuall text, e.g. when predicted link > max_idx, is set to predicted_link== max_idx
        https://arxiv.org/pdf/1704.06104.pdf

        """
        new_links = []
        for i in range(links.shape[0]):
            last_unit_idx = max(0, lengths_units[i]-1)
            sample_links = links[i]

            if decoded:
                sample_links = self.label_encoders["link"].encode_token_links(
                                                                                sample_links,
                                                                                span_token_lengths=span_token_lengths[i],
                                                                                none_spans=none_spans[i]
                                                                                )

            
            links_above_allowed = sample_links > last_unit_idx
            sample_links[links_above_allowed] = last_unit_idx

            links_above_allowed = sample_links < 0
            sample_links[links_above_allowed] = last_unit_idx

            sample_links = self.label_encoders["link"].decode_token_links(
                                                                        sample_links,
                                                                        span_token_lengths=span_token_lengths[i],
                                                                        none_spans=none_spans[i]
                                                                        )
            new_links.append(sample_links)
        
        return np.array(new_links)


    def add_loss(self, task:str, data=torch.tensor):

        assert torch.is_tensor(data), f"{task} loss need to be a tensor"
        #assert data.requires_grad, f"{task} loss tensor should require grads"

        if task == "total":
            self.loss["total"] = data
        else:
            self.loss[task] = data

            if not self._total_loss_added:

                if "total" not in self.loss:
                    self.loss["total"] = data
                else:
                    self.loss["total"] += data
    
        self.metrics.update({f"{task}-loss":int(data) if not torch.isnan(data) else 0})


    def add_preds(self, task:str, level:str, data:torch.tensor, decoded:bool=False, sample_ids="same"):

        #assert task in set(self.tasks), f"{task} is not a supported task. Supported tasks are: {self.tasks}"
        assert level in set(["token", "unit"]), f"{level} is not a supported level. Only 'token' or 'unit' are supported levels"
        assert torch.is_tensor(data) or isinstance(data, np.ndarray), f"{task} preds need to be a tensor or numpy.ndarray"
        assert len(data.shape) == 2, f"{task} preds need to be a 2D tensor"

        data = ensure_numpy(data)
    
        # If we are prediction on ACs and input is on AC level we can use the information of how many ACs a sample has directly from out input.
        # Same applies if prediction level is token; we get this information explicitly from our preprocessing
        # and same applies if input level i token and prediction level is AC
        # However, if the prediction level is token and the input level is AC, we need to use the prediction lengths derived from our segmentation predictions
        # if level == "unit" and self.prediction_level == "unit":
        #     lengths = self.batch["unit"]["lengths"]

        # elif level == "token" and self.prediction_level == "token":
        #     lengths = self.batch["token"]["lengths"]

        # elif level == "token" and self.prediction_level == "unit":
        #     lengths = self.batch["token"]["lengths"]

        # elif level == "unit" and self.prediction_level == "token":
        #     #lengths = self.seg["lenghts_seq"]
        #     raise NotImplementedError()
            

        if "+" in task:
            self.__handle_complex_tasks(
                                        data=data,
                                        level=level,
                                        lengths=ensure_numpy(self.batch[level]["lengths"]),
                                        task=task
                                        )
            return


        if level == "unit":
            if self.prediction_level == "unit":
                span_indexes = ensure_numpy(self.batch["unit"]["span_idxs"])
            else:
                raise NotImplementedError()

            # turn labels for each span to labels across all tokens in sample
            data = self.__unfold_span_labels(
                                            span_labels=data,
                                            span_indexes=span_indexes,
                                            max_nr_token=max(ensure_numpy(self.batch["token"]["lengths"])),
                                            )

            if not self._pred_span_set:
                self.pred_spans["unit_lengths"] = self.batch[level]["lengths"]
                self.pred_spans["lengths_tok"] = self.batch["span"]["lengths_tok"]
                self.pred_spans["none_span_mask"] = self.batch["span"]["none_span_mask"]
                self._pred_span_set = True

            level = "token"


        if task == "link":
            data = self.__correct_links(
                                        data,
                                        lengths_units=ensure_numpy(self.pred_spans["unit_lengths"]),
                                        span_token_lengths=ensure_numpy(self.pred_spans["lengths_tok"]),
                                        none_spans=ensure_numpy(self.pred_spans["none_span_mask"]),
                                        decoded=decoded
                                        )


        if not decoded:

            if task == "link" and level == "token":
                decoded_preds = self.__decode_token_link_labels(  
                                                                    preds=data, 
                                                                    lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                                    span_lengths=ensure_numpy(self.pred_spans["lengths_tok"]),
                                                                    none_spans=ensure_numpy(self.pred_spans["none_span_mask"])
                                                                    )
                if not self.inference:
                    decoded_targets = self.__decode_token_link_labels(
                                                                        preds=ensure_numpy(self.batch[level][task]), 
                                                                        lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                                        span_lengths=ensure_numpy(self.batch["span"]["lengths_tok"]),
                                                                        none_spans=ensure_numpy(self.batch["span"]["none_span_mask"])
                                                                        )

            else:
                decoded_preds = self.__decode_labels(
                                                    preds=data, 
                                                    lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                    task=task
                                                    )
                if not self.inference:
                    decoded_targets = self.__decode_labels(
                                                            preds=ensure_numpy(self.batch[level][task]), 
                                                            lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                            task=task
                                                            )
        else:
            decoded_preds = data

            if task == "link" and level == "token":
                if not self.inference:
                    decoded_targets = self.__decode_token_link_labels(
                                                            preds=ensure_numpy(self.batch[level][task]), 
                                                            lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                            span_lengths=ensure_numpy(self.batch["span"]["lengths_tok"]),
                                                            none_spans=ensure_numpy(self.batch["span"]["none_span_mask"])
                                                            )
            else:
                if not self.inference:
                    decoded_targets = self.__decode_labels(
                                                            preds=ensure_numpy(self.batch[level][task]), 
                                                            lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                            task=task
                                                            )


        if task == "seg":
            bio_data = bio_decode(
                                    batch_encoded_bios=decoded_preds,
                                    lengths=ensure_numpy(self.batch[level]["lengths"]),
                                )
            self.pred_spans["lengths_tok"] = bio_data["span"]["lengths_tok"]
            self.pred_spans["none_span_mask"] = bio_data["span"]["none_span_mask"]
            self.pred_spans["unit_lengths"]  = bio_data["unit"]["lengths"]
        


        mask = ensure_numpy(self.batch["token"]["mask"])
        preds = ensure_flat(ensure_numpy(decoded_preds), mask=mask)
        targets = ensure_flat(ensure_numpy(decoded_targets), mask=mask)
        self.outputs[task] = preds

        if not self.inference:
            self.outputs[f"T-{task}"] = targets
            self.metrics.update(token_metrics(
                                            targets=preds,
                                            preds=targets,
                                            task=task,
                                            labels=self.label_encoders[task].labels,
                                            )
                                )


    def add_probs(self, task:str, level:str, data:torch.tensor):
        raise NotImplementedError
        

    def to_df(self):
        return pd.DataFrame(self.outputs)


    def to_record(self):
        return self.to_df().to_dict("records")


    # def show_sample(self, prefix=""):
        
    #     tokens = []
    #     pred_labels = None
    #     pred_span_lengths = []
    #     pred_span_mask = []

    #     if "label" in self.outputs:
    #         pred_labels = []

    #     for idx in  self.batch.oo:
    #         token_len = self.batch["token"]["lengths"][idx]
    #         tokens.extend(self.batch[self.prediction_level]["text"][idx][:token_len])
            
    #         if pred_labels is not None:
    #             pred_labels.extend(self.outputs["label"][idx][:token_len])

    #         pred_span_lengths.extend(self.pred_spans["lengths_tok"][idx])
    #         pred_span_mask.extend(self.pred_spans["none_span_mask"][idx])

    #     # if "link" in self.outputs:

    #     #     link_labels = self.outputs.get("link_label", None)
    #     #     if link_labels is not None:
    #     #         link_labels = link_labels[idx]


    #     #     links = self.label_encoders["link"].encode_token_links(
    #     #                                                                     self.outputs["link"][idx],
    #     #                                                                     span_token_lengths=self.pred_spans["lengths_tok"][idx],
    #     #                                                                     none_spans=self.pred_spans["none_span_mask"][idx]
    #     #                                                                     )

    #     #     arrays_to_tree(
    #     #                     self.pred_spans["lengths"][idx], 
    #     #                     self.pred_spans["lengths_tok"][idx],
    #     #                     self.pred_spans["none_span_mask"][idx],
    #     #                     links=links,
    #     #                     labels=self.outputs["label"][idx],
    #     #                     tokens=self.batch[self.prediction_level]["text"][idx],
    #     #                     label_colors=get_dataset("pe").label_colors(),
    #     #                     link_labels=link_labels,
    #     #                     )
        
    #     return highlight_text(
    #                     tokens=tokens,
    #                     labels=["Premise", "Claim", "MajorClaim"], 
    #                     pred_labels=pred_labels,
    #                     pred_span_lengths=pred_span_lengths,
    #                     pred_none_spans=pred_span_mask,
    #                     gold_labels=None,
    #                     gold_span_lengths=None,
    #                     gold_none_spans=None,
    #                     save_path=None, 
    #                     return_html=True, 
    #                     prefix=prefix,
    #                     # show_spans:bool=True, 
    #                     show_scores=True if pred_labels is not None else False, 
    #                     # show_legend:bool=True,
    #                     # font:str="Verdana", 
    #                     # width:int=1000, 
    #                     # height:int=800
    #                     )

    
