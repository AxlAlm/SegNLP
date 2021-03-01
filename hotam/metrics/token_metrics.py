

#basics
import numpy as np
import warnings
import os
import pandas as pd
from typing import List, Dict, Union, Tuple
import re
from copy import deepcopy
from collections import Counter


#pytroch
import torch


#sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

#SegeVal
#import segeval

#am 
#from hotam.datasets.base.dataset import DataSet
from hotam.utils import ensure_flat, ensure_numpy
from hotam import get_logger



def token_metrics(targets:np.ndarray, preds:np.ndarray, mask:np.ndarray, task:str, labels:list):
    
    preds = ensure_flat(ensure_numpy(preds), mask=mask)
    targets = ensure_flat(ensure_numpy(targets), mask=mask)

    assert targets.shape == preds.shape, f"shape missmatch for {task}: Targets:{targets.shape} | Preds: {preds.shape}"

    label_counts = Counter({l:0 for l in labels})
    label_counts += Counter(preds.tolist()+targets.tolist())

    if task != "link":
        conf_m = confusion_matrix(targets, preds, labels=labels)

    rs = recall_score(targets, preds, labels=labels, average=None)
    ps = precision_score(targets, preds, labels=labels, average=None)
    
    label_metrics = []
    for i,label in enumerate(labels):

        # if a label is not present in the targets or the predictions we ignore is so it doesn count towards the average
        if label_counts[label] == 0:
            continue

        p = ps[i]
        r = rs[i]
        f1 = 2 * ((p*r) / (p+r))
        
        if np.isnan(f1):
            f1 = 0

        label_metrics.append({"name":f"{task}-{label}-precision", "metric": "precision" , "value":p})
        label_metrics.append({"name":f"{task}-{label}-recall", "metric": "recall", "value":r })
        label_metrics.append({"name":f"{task}-{label}-f1", "metric": "f1", "value": f1})


    df = pd.DataFrame(label_metrics)
    task_metrics = [
                        {"name":f"{task}-precision", "metric": "precision", "value": int(df[df["metric"]=="precision"].mean())},
                        {"name":f"{task}-recall", "metric": "recall", "value": int(df[df["metric"]=="recall"].mean())},
                        {"name":f"{task}-f1", "metric": "f1", "value": int(df[df["metric"]=="f1"].mean())},
                    ]

    df = pd.DataFrame(label_metrics + task_metrics) #.loc[:, ["name", "value"]].to_dict("record")
    
    return df["name"].to_numpy().tolist(), df["value"].to_numpy().tolist()
    











    # def _get_seg_label_metrics(self, output_dict:dict) -> Tuple[list, list, list, list]:   
    #     scores = {}
    #     main_task_values = []
    #     for task in self.dataset.all_tasks:
            
    #         #targets = ensure_flat(ensure_numpy(self.batch[task]), mask=mask)
    #         #targets = self._ensure_numpy(self.batch.get_flat(task, remove=self.dataset.task2padvalue[task])) #.cpu().detach().numpy()  #.cpu().detach().numpy()

    #         # calculates the metrics and the class metrics if wanted
    #         task_scores, task_class_scores = self._get_metrics(targets, output_dict, task)

    #         if task in output_dict["loss"]:
    #             task_scores["-".join([self.split, task, "loss"]).lower()]  = ensure_numpy(output_dict["loss"][task])

    #         scores.update(task_scores)
    #         scores.update(task_class_scores)

    #         # for main tasks we want to know the average score
    #         # so we add these to a list, which we easiliy can turn into an
    #         # DataFrame and average
    #         if task in self.dataset.subtasks:
    #             # comb_task_name = "_".join(self.dataset.tasks).lower()

    #             # #renaming to combined task
    #             #rn_scores = {re.sub(r"-\w+-", f"-{comb_task_name}-", k):v for k,v in task_scores.items() if "confusion_matrix" not in k}
                
    #             rn_scores = {re.sub(r"-\w+-", "-average-", k):v for k,v in task_scores.items() if "confusion_matrix" not in k}
    #             main_task_values.append(rn_scores)
            

    #     if len(self.dataset.tasks) > 1:
    #         average_main_task_scores = pd.DataFrame(main_task_values).mean().to_dict()
    #         scores.update(average_main_task_scores)

    #     return scores



    # def __segmentation_evaluation(self, list_segments):
    #     pass


    # def get_eval_metrics(self, output_dict, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     normalize complex labels, calculates the metrics, creates DataFrames for task and class metrics.

    #     Parameters
    #     ----------
    #     batch_dict : [type]
    #         batch outputs

    #     Returns
    #     -------
    #     List[pd.DataFrame, pd.DataFrame]
    #         DataFrame for task scores and class scores
    #     """
    #     #mask = self.batch[f"{self.batch.prediction_level}_mask"]
    #     #self._complex_label2subtasks(output_dict["preds"], mask)

    #     if self._segmentation:
    #         segmentation_score = self.__segmentation_evaluation(output_dict["segment_lengths"])
        

    #     ### WE NEED DECIDE HOW TO EVALUATE METRICS.
    #     # idea is to compare sampel by sample
    #     # then we can first get labels of segments
    #     #
    #     #
    #     # we could flatten the predictions as well and we add padding in gold for each sample where the predicted segments
    #     # have more ACs than the gold?
    #     #
    #     # Or we can chose to cut it off.
    #         seg_predictions = {}
    #         for task in self.dataset.subtasks:
    #             seg_predictions[task] = self.__get_segment_labels(
    #                                                                 output_dict[task],
    #                                                                 output_dict["segment_lengths"], 
    #                                                                 task
    #                                                                 )
            
    #     else:
    #         # if prediction level is AC and we dont do segmentation 
    #         # we can create the pairs of Argument Components easily.
    #         for task in self.dataset.subtasks:
    #             (output_dict[task], self.batch[task])

        

    #     # now we have the labels for each segment
    #     eval_metrics  = self._get_seg_label_metrics([])

    #     return eval_metrics
