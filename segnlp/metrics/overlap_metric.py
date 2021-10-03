
#basics
import numpy as np
import pandas as pd
from collections import Counter
from pandas.core.frame import DataFrame
from typing import List

#sklearn
from sklearn.metrics import confusion_matrix

#segnlp
from segnlp.utils.overlap import find_overlap


def label_confusion_matrix(
                            labels : list,
                            threshold : float,
                            i2ratio : dict,
                            j2ratio : dict,
                            i2j : dict,
                            j2label : dict,
                            i2label : dict,
                            ) -> np.ndarray:


    """
    Creates a Confusion Matrix for label predictions
    """

    #we add 1 for NO MATCH
    n_labels = len(labels) + 1

    #empty confusion matrix
    cm = np.zeros((n_labels, n_labels))


    # First we look at all the FN
    for j, ratio in j2ratio.items():

        if ratio >= threshold:
            continue
        
        cm[j2label[j], -1] += 1


    # Then we look at FN, TP and FP
    for i, ratio in i2ratio.items():

        # if a predicted segment is below ratio we 
        # dont count it
        if ratio < threshold:
            continue
    
        j = i2j[i]
        target_label = j2label[j]
        pred_label = i2label[i]

        cm[target_label, pred_label] += 1

    return cm


def link_label_confusion_matrix( 
                                labels : list,
                                threshold : float,
                                i2ratio : dict,
                                j2ratio : dict,
                                i2j : dict,
                                j2i : dict,
                                j2jt : dict,
                                i2it : dict,
                                j2link_label : dict,
                                i2link_label : dict,
                               ) -> np.ndarray:

    #we add 1 for NO MATCH
    n_labels = len(labels) + 1

    #empty confusion matrix
    cm = np.zeros((n_labels, n_labels))


    # First we look at all the FN
    for j, ratio in j2ratio.items():
        
        # if there is no i for j, we can count it as a FN
        if j not in j2i:
            cm[j2link_label[j], -1] += 1
            continue

        # make sure the source doesn't match over threshold
        if ratio < threshold:
            cm[j2link_label[j], -1] += 1
            continue     

        # check if the target has a match over the threshold
        jt = j2jt[j]
        if j2ratio[jt] < threshold:
            cm[j2link_label[j], -1] += 1
            continue     

        # # check if the target and the source are truly linked
        ji = j2i[j]
        jit = i2it[ji]
        ijt = i2j[jit]
        if ijt != jt:
            cm[j2link_label[j], -1] += 1


    # Then we look at FN, TP and FP
    for i, ratio in i2ratio.items():

        # if there is no j for i,we counted it as a FN
        # if i not in i2j:
        #     continue

        # check if the source match
        if ratio < threshold:
            continue
        
        # check if the target match
        # j's target
        j = i2j[i]
        jt = j2jt[j]
        if j2ratio[jt] < threshold:
            continue

        # check if the target and the source are truly linked
        jt = j2jt[j]
        it = i2it[i]
        ijt = i2j[it]
        if  ijt != jt:
            continue

        #find target and predicted labels
        target_link_label = j2link_label[j]
        pred_link_label = i2link_label[i]

        cm[target_link_label, pred_link_label] += 1

    return cm


def calc_f1(cm:np.array, labels:list, prefix:str, task:str):
    """

    If we are calculating the F1 for a label the we first calculate the
    TP, FP and FN in the follwing way using the confuson matrix we created.

    For label A:

    # T\P | A | B | C | NO |
    # ----------------------
    # A   | TP| FN| FN| FN |
    #-----------------------
    # B   | FP|   |   |    |
    #-----------------------
    # C   | FP|   |   |    | 
    #-----------------------
    # NO  | FP|   |   |    | 


    For label B

     T\P | A | B | C |  NO|
     ----------------------
     A   |   | FP|   |    |  
     ----------------------
     B   | FN| TP| FN| FN |
     ----------------------
     C   |   | FP|   |    |
     ----------------------
     NO  |   | FP|   |    |  


    then we se the following formula fot the the f1
    
    f1 = (2*TP) / ((2*TP) + FP + FN)


    We then average the f1 across the labels to get the f1-macro
    """

    scores = {}
    
    task_TP = 0
    task_FP = 0
    task_FN = 0

    for i,label in enumerate(labels):

        if task == "label" and label == "None":
            continue

        TP = cm[i,i]
        
        #all predictions of label which where actually other labels (minus "miss")
        FP = sum(cm[:,i]) - TP

        # total miss + 
        FN = sum(cm[i]) - TP
    
        f1 = (2*TP) / ((2*TP) + FP + FN)
        scores[f"{prefix}{label}-f1"] = 0 if np.isnan(f1) else f1

        task_TP += TP
        task_FP += FP
        task_FN += FN

    scores[f"{prefix}f1"] = np.mean(list(scores.values()))

    # When the the metric is used in https://arxiv.org/pdf/1704.06104.pdf,
    # they use the micro f1, as they sum the TP and FP and FNs over all labels
    f1_micro = (2*task_TP) / ((2*task_TP) + task_FP + task_FN)
    scores[f"{prefix}f1-micro"] = 0 if np.isnan(f1_micro) else f1_micro 

    return scores


def overlap_metric(pred_df:pd.DataFrame, target_df:pd.DataFrame, task_labels:dict, thresholds: List[float] = [0.5, 1.0]):

    """
    Metric is from the following paper ( chapt. 6.1 Experimental Setup, p. 1390):

    https://aclanthology.org/N16-1164.pdf
    

    NOTE!
            i = predicted segments
            j = TARGET/ground truth/True segment

            it = target of the predicted segment, i.e. i = source and it = target
            jt = target of the target segment ,  i.e. j = source and jt = target

            ji = the predicted segment i that match with target segment j
            ij = the target segment that match with the predicetd segment i

            ijt = the TARGET of the target segment that match with i
            jit = the TARGET of the predicted segment i that match with j


    """

    calc_lable_metric = "label" in task_labels
    calc_link_lable_metric = "link_label" in task_labels and len(pred_df["target_id"].unique()) != 0

    # we also have information about whether the seg_id is a true segments 
    # and if so, which TRUE segmentent id it overlaps with, and how much
    i2ratio, j2ratio, i2j, j2i = find_overlap(
                                            target_df = target_df,  
                                            pred_df = pred_df
                                            )

    #groupby seg_id
    target_df = target_df.groupby("seg_id", sort = False).first()
    pred_df = pred_df.groupby("seg_id", sort = False).first()

    #update js with stuff we didnt find matches for 
    js = target_df.index.to_numpy()
    j2ratio.update({j:0 for j in js if j not in j2ratio})

    #source to target mappings for target and predicted segments
    j2jt = dict(zip(target_df.index, target_df["target_id"]))
    i2it = dict(zip(pred_df.index, pred_df["target_id"]))

    # link labels
    j2label = dict(zip(target_df.index, target_df["label"].astype(int)))
    j2link_label = dict(zip(target_df.index, target_df["link_label"].astype(int)))

    # link labels
    i2label = dict(zip(pred_df.index, pred_df["label"].astype(int)))
    i2link_label = dict(zip(pred_df.index, pred_df["link_label"].astype(int)))

    assert len(set(j2ratio.keys()).difference(set(j2jt.keys()))) == 0

    
    metrics = {}

    for threshold in thresholds:

        #if calc_lable_metric:

        ### LABEL METRICS
        label_cm = label_confusion_matrix(
                                labels = task_labels["label"],
                                threshold = threshold,
                                i2ratio = i2ratio,
                                j2ratio = j2ratio,
                                i2j = i2j,
                                j2label = j2label,
                                i2label = i2label,
                                )

        label_metrics = calc_f1( 
                        label_cm, 
                        labels = task_labels["label"],
                        prefix =f"label-{threshold}-",
                        task = "label",
                        )

        metrics.update(label_metrics)


        ### LINK LABEL METRICS
        #if calc_link_lable_metric:
        link_label_cm = link_label_confusion_matrix(
                                labels = task_labels["link_label"],
                                threshold = threshold,
                                i2ratio = i2ratio,
                                j2ratio = j2ratio,
                                i2j = i2j,
                                j2i = j2i,
                                j2jt = j2jt,
                                i2it = i2it,
                                j2link_label = j2link_label,
                                i2link_label = i2link_label,
                                )

        link_label_metrics = calc_f1( 
                        link_label_cm, 
                        labels = task_labels["link_label"],
                        prefix = f"link_label-{threshold}-",
                        task = "link_label"
                        )
        metrics.update(link_label_metrics)


        metrics[f"{threshold}-f1"] = np.mean([
                                                metrics[f"link_label-{threshold}-f1"],
                                                metrics[f"label-{threshold}-f1"],
                                            ])

        metrics[f"{threshold}-f1-micro"] = np.mean([
                                                metrics[f"link_label-{threshold}-f1-micro"],
                                                metrics[f"label-{threshold}-f1-micro"],
                                            ])

    return metrics
