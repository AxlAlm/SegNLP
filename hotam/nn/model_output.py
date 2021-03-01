

#basics
from typing import Union, List
import numpy as np


#hotam
from hotam.metrics import token_metrics
from hotam.nn import ModelInput
from hotam.utils import ensure_numpy

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
                calc_metrics:bool=True,
                ):

        self.return_output = return_output
        self.calc_metrics = calc_metrics
        self.label_encoders = label_encoders
        self.tasks = tasks
        self.all_tasks = all_tasks
        self.prediction_level = prediction_level
        self.batch = batch
        self._total_loss_added = False

        self.loss = {}
        self.metrics_keys = []
        self.metric_values = []
        self.outputs = []
        
        # self.dataset.get_subtask_position(task, subtask) 
        # #self.dataset.decode_list(preds[i][:lenghts[i]], task)
        # self.dataset.prediction_level
        # self.dataset.all_tasks
        # self.dataset = dataset
        #self._need_seg_lens = if self.dataset.prediction_level == "token"
        #self._seg_lens_added = False

        # if "seg" in self.dataset.subtasks:
        #     seg_task = [t for t in self.dataset.tasks if "seg" in task][0]
        #     Bs = [i for l,i in self.dataset.encoders[seg_task].label2id.items() if l.lower().startswith("b")]
        #     Is = [i for l,i in self.dataset.encoders[seg_task].label2id.items() if l.lower().startswith("i")]
        #     Os = [i for l,i in self.dataset.encoders[seg_task].label2id.items() if l.lower().startswith("o")]
        #     self.bio = BIO_Decoder(
        #                             Bs=Bs,
        #                             Is=Is,
        #                             Os=Os,
        #                             )
        

        #self.seg = {}  
    @property
    def metrics(self):
        return self.metrics_keys, np.array(self.metric_values)


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
    

    def __decode_labels(self, preds:list, lengths:list, task:str):

        dtype = "<U30" if task != "link" else np.int16
        size = preds.shape[0]
        decoded_preds = np.zeros((size, max(lengths)), dtype=dtype)
        for i in range(size):
            decoded_preds[i][:lengths[i]] = self.label_encoders[task].decode_list(preds[i][:lengths[i]])
            #decoded_preds[i][:lengths[i]] = self.dataset.decode_list(preds[i][:lengths[i]], task)
        
        return decoded_preds


    def __handle_complex_tasks(self, data:torch.tensor, level:str, lengths:list, task:str):

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

        tok_labels = np.zeros((span_labels.shape[0], max_nr_token))
        for i in range(batch_size):
            for start,end in span_indexes[i]:
                tok_labels[i][start:end] = span_labels[i][j]
        return tok_labels
        

    def __correct_segmentation(self, data):
        pass

    
    def __correct_relations(self, data):
        pass


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
    
        self.metrics_keys.append(f"{task}-loss")
        self.metric_values.append(int(data))


    def add_preds(self, task:str, level:str, data:torch.tensor, decoded:bool=False, sample_ids="same"):

        #assert task in set(self.tasks), f"{task} is not a supported task. Supported tasks are: {self.tasks}"
        assert level in set(["token", "span"]), f"{level} is not a supported level. Only 'token' or 'span' are supported levels"
        assert torch.is_tensor(data) or isinstance(data, np.ndarray), f"{task} preds need to be a tensor or numpy.ndarray"
        assert len(data.shape) == 2, f"{task} preds need to be a 2D tensor"

    
        # If we are prediction on ACs and input is on AC level we can use the information of how many ACs a sample has directly from out input.
        # Same applies if prediction level is token; we get this information explicitly from our preprocessing
        # and same applies if input level i token and prediction level is AC
        #
        # However, if the prediction level is token and the input level is AC, we need to use the prediction lengths derived from our segmentation predictions
        if level == "span" and self.prediction_level == "span":
            lengths = self.batch["lengths_seq"]

        elif level == "token" and self.prediction_level == "token":
            lengths = self.batch["lengths_tok"]

        elif level == "token" and self.prediction_level == "span":
            lengths = self.batch["lengths_tok"]

        elif level == "span" and self.prediction_level == "token":
            #lengths = self.seg["lenghts_seq"]
            raise NotImplementedError()


        if "+" in task:
            print(task, data)
            self.__handle_complex_tasks(
                                        data=data,
                                        level=level,
                                        lengths=lengths,
                                        task=task
                                        )
            return


        if level == "span":
            if self.prediction_level == "span":
                span_indexes = self.batch["span_idxs"]
            else:
                raise NotImplementedError()

            # turn labels for each span to labels across all tokens in sample
            data = self.__unfold_span_labels(
                                            ac_labels=data,
                                            span_indexes=span_indexes,
                                            max_nr_token=max(self.batch["lengths_tok"]),
                                        )

        if not decoded:
            decoded_preds = self.__decode_labels(
                                                preds=data, 
                                                lengths=self.batch["lengths_tok"],
                                                task=task
                                                )
        else:
            decoded_preds = data


        # if task == "seg":
        #     decoded_labels = self.__correct_segmentation(decoded_labels)

        # if task == "link":
        #     decoded_labels = self.__correct_relations(decoded_labels)

        # print(task)
        # print("DECODED PREDS", decoded_labels)
        # print("TARGETS", self.batch[f"token_{task}"])
        if self.calc_metrics:
            #self.batch[f"token_{task}"]
            decoded_targets = self.__decode_labels(
                                                preds=self.batch[f"token_{task}"], 
                                                lengths=self.batch["lengths_tok"],
                                                task=task
                                                )

            keys, values = token_metrics(
                                                targets=decoded_targets,
                                                preds=decoded_preds,
                                                mask=self.batch["token_mask"],
                                                task=task,
                                                labels=self.label_encoders[task].labels,
                                                )
            self.metrics_keys.extend(keys)
            self.metric_values.extend(values)

   
                            

        
        # if return_output:


        #     if sample_ids == "same":

        #         if isinstance(return_output) == bool:

        #             set(ensure_numpy(self.batch["ids"]).tolist()).union(set(return_output))

        #     else:
        #         raise NotImplementedError()


        # #self.preds[task] = seg_labels


    def add_probs(self, task:str, level:str, data:torch.tensor):

        assert torch.is_tensor(data), f"{task} probs need to be a tensor"
        assert len(data.shape) == 3, f"{task} probs need to be a 3D tensor"
        
        pass
        

    # def clear(self):
    #     self.batch = None
    #     self.loss = {}
    #     self.output = {}







    # def __get_segment_labels(labels:list, seg_lengths:list, seg_types:list, max_nr_segs:int, task:str):

    #     nr_samples = labels.shape[0]
    #     seg_labels = torch.zeros(nr_samples, max_nr_segs)
    #     i = 0
    #     for i in range(nr_samples):
    #         floor = 0
    #         type_length = zip(seg_types[i] , seg_lengths[i])
    #         # we dont want segments taht are not Argument Components. These ones we filter out.
    #         sample_seg_lens = [l for t,l in type_length if t is not None] 
    #         nr_segs = len(sample_seg_lens)
    #         for j in range(nr_segs):
                    
    #             seg_labels = labels[i][floor:floor+length]
    #             most_freq_label = Counter(seg_labels).most_common(1)[0][0]

    #             if task == "relation":
                    
    #                 point_to_idx = j + most_freq_label 
    #                 if point_to_idx > nr_segs or point_to_idx < 0:
    #                     most_freq_label = nr_segs - j

    #             seg_labels[i][j] = most_freq_label
    #             floor += length

    #     return seg_labels





    # def __handle_segmentation(self,data);
    #     self.seg["seg_lengths"], self.seg["types"], self.seg["lenghts_seq"] = self.bio.decode(
    #                                                                                         batch_encoded_bios=data, 
    #                                                                                         lengths=self.batch["lengths_tok"]
    #                                                                                         )

    #     # if calc_metrics:
    #     #     seg_results = calc_seg_metrics( 
    #     #                                     target_seg_lens=self.batch["lengths_seq"], 
    #     #                                     pred_seg_lens=self.seg["seg_lengths"]
    #     #                                     )

    #     # self._seg_lens_added = True

    # def add_preds(self, task:str, level:str, data:torch.tensor):

    #     assert task in set(self.dataset.all_tasks), f"{task} is not a supported task. Supported tasks are: {self.dataset.all_tasks}"
    #     assert level in set(["token", "ac"]), f"{level} is not a supported level. Only 'token' or 'ac' are supported levels"
    #     assert torch.is_tensor(data), f"{task} preds need to be a tensor"
    #     assert len(data.shape) == 2, f"{task} preds need to be a 2D tensor"
        
    #     if "seg" in task and not self._seg_lens_added:
    #         self.__handle_segmentation(
    #                                     data=data
    #                                     )
    #         return


    #     # If we are prediction on token level we need to convert the labels to segment labels. For this we need
    #     # the length of each segment predicted by the model.
    #     if level == "token":
    #         if not self._seg_lens_added:
    #             raise RuntimeError("When segmentation is a subtasks it needs to be added first to output")
    
    #     # If we are prediction on ACs and input is on AC level we can use the information of how many ACs a sample has directly from out input.
    #     # Same applies if prediction level is token; we get this information explicitly from our preprocessing
    #     # and same applies if input level i token and prediction level is AC
    #     #
    #     # However, if the prediction level is token and the input level is AC, we need to use the prediction lengths derived from our segmentation predictions
    #     if level == "ac" and self.dataset.prediction_level == "ac":
    #         lengths = self.batch["lengths_seq"]

    #     elif level == "token" and self.dataset.prediction_level == "token":
    #         lengths = self.batch["lengths_tok"]

    #     elif level == "ac" and self.dataset.prediction_level == "token":
    #         lengths = self.seg["lenghts_seq"]

    #     elif level == "token" and self.dataset.prediction_level == "ac":
    #         lengths = self.batch["lengths_tok"]


    #     if "_" in task:
    #         self.__handle_complex_tasks(
    #                                     data=data,
    #                                     level=level,
    #                                     lengths=lengths
    #                                     )
    #         return



    #     if level == "ac":
    #         seg_labels = self.__decode_labels(
    #                                             preds=data, 
    #                                             lengths=lengths
    #                                             )

    #     else:
    #         decoded_labels = self.__decode_labels(
    #                                                 preds=data, 
    #                                                 lengths=lengths
    #                                                 )

    #         seg_labels = self.__get_segment_labels(
    #                                                 labels=decoded_labels, 
    #                                                 seg_lengths=,
    #                                                 seg_types=list,
    #                                                 max_nr_segs=self.dataset.max_nr_segs, 
    #                                                 task=task
    #                                                 )
            
    #     if calc_metrics:

    #         if self.dataset.prediction_level == "ac":
    #             pred_lengths = self.batch["lengths_seq"]
    #         else:
    #             pred_lengths = self.seg["lengths_seq"]

    #         task_metrics = calc_metrics(
    #                                     targets=self.batch["AC_TARGETS FOR TASK"],
    #                                     preds=seg_labels,
    #                                     target_lengths=self.batch["lengths_seq"],
    #                                     pred_lengths=pred_lengths,
    #                                     task=task,
    #                                     prefix=self.current_split,
    #                                     )

    #     self.preds[task] = seg_labels




    # outputs = {}

    # #first we decode the complex tasks, and get the predictions for each of the subtasks
    # complex_tasks = [task for task in self.dataset.tasks if "_" in task]
    # for c_task in complex_tasks:
    #     decoded_labels = decode(preds=preds, lengths=lengths)
    #     subtask_preds = get_subtask_preds(decoded_labels=decoded_labels, task=c_task)
    #     output_dict.update(subtask_preds)


    # for task in self.dataset.subtasks:
        
    #     #if we have the decoded labels
    #     if task not in output_dict:
    #         decoded_labels = outputs[task]
        
    #     #if we dont have the decoded labels
    #     else:
    #         decoded_labels = decode(preds=preds, lengths=lengths)
    #         outputs[task] = decoded_labels


    #     get_segment_labels(decoded_labels, segment_lengths, segment_types, max_nr_segs, task)


    # return outputs
    