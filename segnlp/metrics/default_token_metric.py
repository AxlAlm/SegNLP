

#basics
import pandas as pd

#segnlp
from .metric_utils import f1_precision_recall


def default_token_metric(pred_df:pd.DataFrame, target_df:pd.DataFrame,  task_labels:dict):

    collected_scores = {}
    
    for task in task_labels.keys():

        targets = target_df[task].to_numpy().astype(int)
        preds = pred_df[task].to_numpy().astype(int)

        scores = f1_precision_recall(
                            targets = targets,
                            preds = preds,
                            task = task,
                            labels = task_labels[task],
                            )
        collected_scores.update(scores)
    
    return scores

    
