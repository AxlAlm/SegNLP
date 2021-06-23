

#basics
import pandas as pd

#segnlp
from .metric_utils import f1_precision_recall


def default_token_metric(df:pd.DataFrame, task_labels:dict, task_label_ids:dict):

    collected_scores = {}
    for task in task_labels.keys():

        targets = df[f"T-{task}"].to_numpy()
        preds = df[task].to_numpy()

        scores = f1_precision_recall(
                            targets = targets,
                            preds = preds,
                            task = task,
                            labels = task_labels[task],
                            label_ids = task_label_ids[task]
                            )
        collected_scores.update(scores)
    
    return scores

    
