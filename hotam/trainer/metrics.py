

#basics
import numpy as np
import warnings
import os
import pandas as pd
from typing import List, Dict, Union, Tuple
import re
from copy import deepcopy

#pytroch
import torch


#sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#am 
from hotam.preprocessing import DataSet
from hotam.utils import ensure_flat, ensure_numpy
from hotam import get_logger


class Metrics:
    
    @classmethod
    def metrics(self):
        metrics = [
                        { 
                            "name": "precision",
                            "function": precision_score,
                            "args": {"average":"macro"},
                            "probs": False,
                            "per_class": True
                        },
                        { 
                            "name": "recall",
                            "function": recall_score,
                            "args": {"average":"macro"},
                            "probs": False,
                            "per_class": True

                        },
                        { 
                            "name": "f1",
                            "function": f1_score,
                            "args": {"average":"macro"},
                            "probs": False,
                            "per_class": True
                        },
                        { 
                            "name": "confusion_matrix",
                            "function": confusion_matrix,
                            "args": {},
                            "probs": False,
                            "per_class": False
                        },
                        # { 
                        #     "name": "auc-roc",
                        #     "function": roc_auc_score,
                        #     "args": {"average":"macro"},
                        #     "probs": False,
                        #     "per_class": False,
                        # },
                        # { 
                        #     "name": "accuracy",
                        #     "function": accuracy_score,
                        #     "args": {},
                        #     "probs": False,
                        #     "per_class": False
                        # },
                        ]

        return metrics, [m["name"] for m in metrics], [m["name"] for m in metrics if m["per_class"]]
  
    def _get_progress_bar_metrics(self, log:dict) -> dict:
        """
        will extract the metrics which should be monitored either by progress bar or callbacks, 
        and return them in a dict

        Parameters
        ----------
        log : dict
            logging dict

        Returns
        -------
        dict
            dict of metrics scores
        """

        current_split = log["split"]
        scores = {}
        for task_metric in [self.monitor_metric]+self.progress_bar_metrics:
            task, metric = task_metric.rsplit("_",1)
            split = task.split("_", 1)

            if current_split != split:
                continue

            scores[task_metric] = log["scores"].loc[task][metric]

        return scores


    def _complex_label2subtasks(self, output_dict:dict, mask:np.ndarray):
        """
        given that some labels are complexed, i.e. 2 plus tasks in one, we can break these apart and 
        get the predictions for each of the task so we can get metric for each of them. 

        for example, if one task is Segmentation+Arugment Component Classification, this task is actually two
        tasks hence we can break the predictions so we cant get scores for each of the tasks.

        this fucntions simple extracts the predictions for any subtask in any complex task

        Parameters
        ----------
        output_dict : dict
            predictions per task from the batch
        """

        def get_subtask_preds(task, preds):
            subtasks_predictions = {}

            # e.g. "seg_ac"
            original_shape = preds.shape
            flat_preds = ensure_flat(ensure_numpy(preds))
            decoded_preds = self.dataset.decode_list(flat_preds, task)
            for subtask in task.split("_"):

                # get the positon of the subtask in the joint label
                # e.g. for  labels following this, B_MajorClaim, 
                # BIO is at position 0, and AC is at position 1
                subtask_position = self.dataset.get_subtask_position(task, subtask)

                # remake joint labels to subtask labels
                subtask_preds = torch.LongTensor(self.dataset.encode_list([p.split("_")[subtask_position] for p in decoded_preds], subtask))
                subtasks_predictions[subtask] = subtask_preds.view(original_shape)

            return subtasks_predictions

        for task, preds in output_dict.copy().items():

            if "_" not in task:
                continue
            
            #ensure_flat(ensure_numpy(preds), mask=mask)
            subtask_preds = get_subtask_preds(task, preds)
            output_dict.update(subtask_preds)
           

    def _get_class_metrics(self, metric:dict,  targets:np.ndarray, preds:np.ndarray, task:str, split:str):

        class_scores_dict = {}

        task_labels_ids = self.dataset.encoders[task].ids
        metric_args = metric["args"].copy()
        metric_args["labels"] = task_labels_ids
        metric_args["average"] = None

        class_scores = metric["function"](targets, preds, **metric_args) #, **new_args)

        for label_id, value in zip(task_labels_ids, class_scores):
            label_name = self.dataset.decode(label_id, task)

            if not isinstance(label_name, str):
                label_name = str(label_name)

            class_score_name = "-".join([split, task, label_name, metric["name"]]).lower()
            class_scores_dict[class_score_name] = value
        
        return class_scores_dict


    def _get_metrics(self, targets:np.ndarray, output_dict:dict, task:str, mask:np.ndarray):
        
        #TODO: change this function to so that it calcualtes the f1 label wise then aggregates to f1 task wise.
        scores = {} 
        class_scores = {}
        for metric in self.metrics:

            if metric["name"] == "confusion_matrix" and "relation" in task:
                continue

            #decide if we want probabilites or predictions for the metric
            if metric["probs"]:
                preds = output_dict["probs"][task]
            else:
                preds = output_dict["preds"][task]

            preds = ensure_flat(ensure_numpy(preds), mask=mask)

            assert targets.shape == preds.shape, f"shape missmatch for {task}: Targets:{targets.shape} | Preds: {preds.shape}"

            metric_args = deepcopy(metric["args"])
            
            # using labels will make the f1 behave a bit odd. If labels that are not present will be counter as 0 in the average calculation
            # we only use it in confusion matrix to ensure we get a same dim tabel
            if metric["name"] == "confusion_matrix":
                metric_args["labels"] = self.dataset.encoders[task].ids

            score_name = "-".join([self.split, task, metric["name"]]).lower()
            score = metric["function"](targets, preds, **metric_args)
            scores[score_name] = score

            if metric["per_class"] and not metric["probs"] and task != "relation":
                class_metric = self._get_class_metrics(metric, targets, preds, task, self.split)
                class_scores.update(class_metric)
        
        return scores, class_scores
            

    def _get_score_log(self, output_dict:dict, mask:np.ndarray) -> Tuple[list, list, list, list]:
        """
        calculates the score for each metrics for each task and for each class if supported. Also calculates the mean metric scores for 
        the main tasks.

        scores are structured in such a way that it can be easy loaded into a multiindex DataFrame.
        
        Parameters
        ----------
        output_dict : dict
            outputs from the batch

        Returns
        -------
        Tuple[list, list, list, list]
            returns values and index for class scores and task scores
        """
        scores = {}
        main_task_values = []
        for task in self.dataset.all_tasks:
            
            targets = ensure_flat(ensure_numpy(self.batch[task]), mask=mask)

            #targets = self._ensure_numpy(self.batch.get_flat(task, remove=self.dataset.task2padvalue[task])) #.cpu().detach().numpy()  #.cpu().detach().numpy()

            # calculates the metrics and the class metrics if wanted
            task_scores, task_class_scores = self._get_metrics(targets, output_dict, task, mask)

            if task in output_dict["loss"]:
                task_scores["-".join([self.split, task, "loss"]).lower()]  = ensure_numpy(output_dict["loss"][task])

            scores.update(task_scores)
            scores.update(task_class_scores)

            # for main tasks we want to know the average score
            # so we add these to a list, which we easiliy can turn into an
            # DataFrame and average
            if task in self.dataset.tasks:
                comb_task_name = "_".join(self.dataset.tasks).lower()

                #renaming to combined task
                rn_scores = {re.sub(r"-\w+-", f"-{comb_task_name}-", k):v for k,v in task_scores.items() if "confusion_matrix" not in k}

                main_task_values.append(rn_scores)
            

        if len(self.dataset.tasks) > 1:
            average_main_task_scores = pd.DataFrame(main_task_values).mean().to_dict()
            scores.update(average_main_task_scores)

        return scores


    def score(self, batch, output_dict, split) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        normalize complex labels, calculates the metrics, creates DataFrames for task and class metrics.

        Parameters
        ----------
        batch_dict : [type]
            batch outputs

        Returns
        -------
        List[pd.DataFrame, pd.DataFrame]
            DataFrame for task scores and class scores
        """
        self.batch = batch 
        self.split = split 
        mask = self.batch[f"{self.batch.prediction_level}_mask"]

        self._complex_label2subtasks(output_dict["preds"], mask)
        scores = self._get_score_log(output_dict, mask)
        return scores
