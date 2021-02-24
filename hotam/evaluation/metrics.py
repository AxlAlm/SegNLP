

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
from sklearn.metrics import confusion_matrix

#SegeVal
#import segeval

#am 
#from hotam.datasets.base.dataset import DataSet
from hotam.utils import ensure_flat, ensure_numpy
from hotam import get_logger



# def calc_seg_metrics(target_seg_lens:np.ndarray, pred_seg_lens:np.ndarray):

#     scores = Counter(dict(""))
#     zipped = zip(target_seg_lens, pred_seg_lens)
#     size = len(target_seg_lens)
#     for target, pred in zipped:
#         conf_matrix = segeval.boundary_confusion_matrix(hypo,ref)
#         segeval.precision(conf_matrix)
#         segeval.recall(conf_matrix)
#         segeval.fmeasure(conf_matrix)

#     scores  = scores / size

#     return scores



def calc_metrics(targets:np.ndarray, preds:np.ndarray, mask:np.ndarray, task:str, prefix:str):

    preds = ensure_flat(ensure_numpy(preds), mask=mask)
    targets = ensure_flat(ensure_numpy(targets), mask=mask)

    assert targets.shape == preds.shape, f"shape missmatch for {task}: Targets:{targets.shape} | Preds: {preds.shape}"

    label_counts = Counter(preds+targets)
    labels = self.dataset.encoders[task].labels
    label_counts = Counter({l:0 for l in labels})
    label_counts += Counter(preds+targets)

    if task != "relation":
        confusion_matrix = confusion_matrix(targets, preds, label=labels)

    rs = recall_score(targets, preds, label=labels)
    ps = precision_score(targets, preds, label=labels)

    label_scores = []
    for i,label in enumerate(labels):

        # if a label is not present in the targets or the predictions we ignore is so it doesn count towards the average
        if label_counts[label] == 0:
            continue

        p = ps[i]
        r = rs[i]
        f1 = 2 * ((p*r) / (p+r))

        rows.append({"name":f"{prefix}-{label}-precision", "metric": "precision" , "value":p})
        rows.append({"name":f"{prefix}-{label}-recall", "metric": "recall", "value":r })
        rows.append({"name":f"{prefix}-{label}-f1", "metric": "f1", "value": f1})


    df = pd.DataFrame(rows)
    avrg_scores = [
                        {"name":f"{prefix}-{task}-precision", "metric": "precision", "value": df[df["metric"]=="precision"].mean()},
                        {"name":f"{prefix}-{task}-recall", "metric": "recall", "value": df[df["metric"]=="recall"].mean()},
                        {"name":f"{prefix}-{task}-f1", "metric": "f1", "value": df[df["metric"]=="f1"].mean()},
                    ]

    scores = df.to_dict("record") + avrg_metrics


    return scores
    











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