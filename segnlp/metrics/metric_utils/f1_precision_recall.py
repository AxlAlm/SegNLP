
#basics
import numpy as np
import pandas as pd
from collections import Counter

#sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def f1_precision_recall(targets:np.ndarray, preds:np.ndarray, task:str, labels:list):

    assert targets.shape == preds.shape, f"shape missmatch for {task}: Targets:{targets.shape} | Preds: {preds.shape}"

    label_counts = Counter({l:0 for l in labels})
    label_counts += Counter(preds.tolist()+targets.tolist())

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
    task_pr = float(df[df["metric"]=="precision"].mean())
    task_re = float(df[df["metric"]=="recall"].mean())
    task_f1 = float(df[df["metric"]=="f1"].mean())

    task_metrics = [
                        {"name":f"{task}-precision", "metric": "precision", "value": task_pr},
                        {"name":f"{task}-recall", "metric": "recall", "value": task_re},
                        {"name":f"{task}-f1", "metric": "f1", "value": task_f1},
                    ]


    #this is added so that the task is taken into account when calculating the mean accross all tasks
    mean_metrics = [
                        {"name":f"precision", "metric": "precision", "value": task_pr},
                        {"name":f"recall", "metric": "recall", "value": task_re},
                        {"name":f"f1", "metric": "f1", "value": task_f1},
                    ]

    df = pd.DataFrame(label_metrics + task_metrics + mean_metrics) #.loc[:, ["name", "value"]].to_dict("record")
    
    return dict(zip(df["name"].to_numpy().tolist(), df["value"].to_numpy().tolist()))