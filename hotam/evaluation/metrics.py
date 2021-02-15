

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
                            "args": {}, #{"average":"macro"},
                            "probs": False,
                            "per_class": True
                        },
                        { 
                            "name": "recall",
                            "function": recall_score,
                            "args": {}, #{"average":"macro"},
                            "probs": False,
                            "per_class": True

                        },
                        { 
                            "name": "f1",
                            "function": f1_score,
                            "args": {}, #{"average":"macro"},
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
  
    # def _get_progress_bar_metrics(self, log:dict) -> dict:
    #     """
    #     will extract the metrics which should be monitored either by progress bar or callbacks, 
    #     and return them in a dict

    #     Parameters
    #     ----------
    #     log : dict
    #         logging dict

    #     Returns
    #     -------
    #     dict
    #         dict of metrics scores
    #     """

    #     current_split = log["split"]
    #     scores = {}
    #     for task_metric in [self.monitor_metric]+self.progress_bar_metrics:
    #         task, metric = task_metric.rsplit("_",1)
    #         split = task.split("_", 1)

    #         if current_split != split:
    #             continue

    #         scores[task_metric] = log["scores"].loc[task][metric]

    #    return scores


    # def _get_class_metrics(self, metric:dict,  targets:np.ndarray, preds:np.ndarray, task:str, split:str):

    #     class_scores_dict = {}

    #     metric_args = metric["args"].copy()
    #     metric_args["labels"] = self.dataset.encoders[task].labels
    #     metric_args["average"] = None

    #     class_scores = metric["function"](targets, preds, **metric_args) #, **new_args)

    #     for label, value in zip(task_labels_ids, class_scores):
    #         class_score_name = "-".join([split, task, label, metric["name"]]).lower()
    #         class_scores_dict[class_score_name] = value
        
    #     return class_scores_dict


    def calc_seg_metrics():
        pass


    def calc_metrics(self, preds:np.ndarray, targets:np.ndarray, mask:np.ndarray, task:str, prefix:str):

        score_name = "-".join([prefix, task, metric["name"]]).lower()
        
        preds = ensure_flat(ensure_numpy(preds), mask=mask)
        targets = ensure_flat(ensure_numpy(targets), mask=mask)
        assert targets.shape == preds.shape, f"shape missmatch for {task}: Targets:{targets.shape} | Preds: {preds.shape}"

        metric_args = deepcopy(metric["args"])
        

        results = {}

        label_counts = Counter(preds+targets)
        labels = self.dataset.encoders[task].labels
        label_counts = Counter({l:0 for l in labels})
        label_counts += Counter(preds+targets)

        if task != "relation":
            confusion_matrix = confusion_matrix(targets, preds, label=labels)

        #scores["f1"] = f1_score(targets, preds, label=self.dataset.encoders[task].labels)
        rs = recall_score(targets, preds, label=labels)
        ps = precision_score(targets, preds, label=labels)

        rows = []
        for i,label in enumerate(labels):
            p = ps[i]
            r = rs[i]
            f1 = 2 * ((p*r) / (p+r))

            rows.append({"name":f"{prefix}-{label}-precision", "metric": "precision" , "value":p})
            rows.append({"name":f"{prefix}-{label}-recall", "metric": "recall", "value":r })
            rows.append({"name":f"{prefix}-{label}-f1", "metric": "f1", "value": f1})


        df = pd.DataFrame(rows)

        #FILTER OUT ZERO COUNTS
        rows_labels_not_present = ""
        avrg_precision = df[df["metric"]=="precision"].mean()
        avrg_recall = df[df["metric"]=="recall"].mean()
        avrg_f1 = df[df["metric"]=="f1"].mean()

        




            # if metric["per_class"] and not metric["probs"]:
            #     class_metric = self._get_class_metrics(metric, targets, preds, task, self.split)
            #     class_scores.update(class_metric)
        
        return scores, class_scores











    def _get_seg_label_metrics(self, output_dict:dict) -> Tuple[list, list, list, list]:   
        scores = {}
        main_task_values = []
        for task in self.dataset.all_tasks:
            
            #targets = ensure_flat(ensure_numpy(self.batch[task]), mask=mask)
            #targets = self._ensure_numpy(self.batch.get_flat(task, remove=self.dataset.task2padvalue[task])) #.cpu().detach().numpy()  #.cpu().detach().numpy()

            # calculates the metrics and the class metrics if wanted
            task_scores, task_class_scores = self._get_metrics(targets, output_dict, task)

            if task in output_dict["loss"]:
                task_scores["-".join([self.split, task, "loss"]).lower()]  = ensure_numpy(output_dict["loss"][task])

            scores.update(task_scores)
            scores.update(task_class_scores)

            # for main tasks we want to know the average score
            # so we add these to a list, which we easiliy can turn into an
            # DataFrame and average
            if task in self.dataset.subtasks:
                # comb_task_name = "_".join(self.dataset.tasks).lower()

                # #renaming to combined task
                #rn_scores = {re.sub(r"-\w+-", f"-{comb_task_name}-", k):v for k,v in task_scores.items() if "confusion_matrix" not in k}
                
                rn_scores = {re.sub(r"-\w+-", "-average-", k):v for k,v in task_scores.items() if "confusion_matrix" not in k}
                main_task_values.append(rn_scores)
            

        if len(self.dataset.tasks) > 1:
            average_main_task_scores = pd.DataFrame(main_task_values).mean().to_dict()
            scores.update(average_main_task_scores)

        return scores



    def __segmentation_evaluation(self, list_segments):
        pass


    def get_eval_metrics(self, output_dict, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        #mask = self.batch[f"{self.batch.prediction_level}_mask"]
        #self._complex_label2subtasks(output_dict["preds"], mask)

        if self._segmentation:
            segmentation_score = self.__segmentation_evaluation(output_dict["segment_lengths"])
        

        ### WE NEED DECIDE HOW TO EVALUATE METRICS.
        # idea is to compare sampel by sample
        # then we can first get labels of segments
        #
        #
        # we could flatten the predictions as well and we add padding in gold for each sample where the predicted segments
        # have more ACs than the gold?
        #
        # Or we can chose to cut it off.
            seg_predictions = {}
            for task in self.dataset.subtasks:
                seg_predictions[task] = self.__get_segment_labels(
                                                                    output_dict[task],
                                                                    output_dict["segment_lengths"], 
                                                                    task
                                                                    )
            
        else:
            # if prediction level is AC and we dont do segmentation 
            # we can create the pairs of Argument Components easily.
            for task in self.dataset.subtasks:
                (output_dict[task], self.batch[task])

        

        # now we have the labels for each segment
        eval_metrics  = self._get_seg_label_metrics([])

        return eval_metrics
