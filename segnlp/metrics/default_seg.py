

#basics
import pandas as pd

#segnlp
from .metrics import f1_precision_recall
from segnlp.metrics.base_token import base_token_metric

def unit_metric(self, df:pd.DataFrame, tasks_labels:dict):
    df = df.groupby("unit_id").first()

    collected_scores = {}
    for task, labels in tasks_labels.items():

        targets = df[f"T-{task}"].to_numpy()
        preds = df[task].to_numpy()

        scores = f1_precision_recall(
                            targets = targets
                            preds = preds,
                            task = task 
                            labels = labels
                            )
        collected_scores.update(scores)
    
    return scores