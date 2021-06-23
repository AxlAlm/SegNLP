

#basics
import pandas as pd

#segnlp
from .metric_utils import f1_precision_recall
from .metric_utils import link_f1


def default_segment_metric(df:pd.DataFrame, task_labels:dict, task_label_ids:dict):
    
    df = df.groupby("T-seg_id").first()


    collected_scores = {}
    for task in task_labels.keys():

        if task == "link":            
            scores = link_f1(
                            targets = [s[f"T-{task}"].tolist() for _, s in df.groupby("sample_id")],
                            preds = [s[task].tolist() for _, s in df.groupby("sample_id")],
                            )
            collected_scores.update(scores)

        else:
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
    
    return collected_scores