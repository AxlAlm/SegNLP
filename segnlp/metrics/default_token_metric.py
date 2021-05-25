

#basics
import pandas as pd

#segnlp
from .metrics import f1_precision_recall


def default_token_metric(self, df:pd.DataFrame, tasks_labels:dict):

    collected_scores = {}
    for task, labels in tasks_labels.items():

        targets = df[f"T-{task}"].to_numpy()
        preds = df[task].to_numpy()

        scores = f1_precision_recall(
                            targets = targets,
                            preds = preds,
                            task = task,
                            labels = labels,
                            )
        collected_scores.update(scores)
    
    return scores

    
