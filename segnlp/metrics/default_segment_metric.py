
#basics
import pandas as pd

#segnlp
from .metric_utils import f1_precision_recall
from .metric_utils import link_f1


def default_segment_metric(pred_df:pd.DataFrame, target_df:pd.DataFrame,  task_labels:dict):
    
    target_df = target_df.groupby("seg_id").first()
    pred_df = pred_df.groupby("seg_id").first()

    collected_scores = {}
    for task in task_labels.keys():

        if task == "link":            
            scores = link_f1(
                            targets = [s[task].tolist() for _, s in target_df.groupby("sample_id", sort = False)],
                            preds = [s[task].tolist() for _, s in pred_df.groupby("sample_id", sort = False)],
                            )
            collected_scores.update(scores)

        else:

            targets = target_df[task].to_numpy()
            preds = pred_df[task].to_numpy()

            scores = f1_precision_recall(
                                        targets = targets,
                                        preds = preds,
                                        task = task,
                                        labels = task_labels[task],
                                        )
            collected_scores.update(scores)
    
    return collected_scores