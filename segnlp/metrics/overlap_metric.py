
#basics
import numpy as np
import pandas as pd
from collections import Counter
from pandas.core.frame import DataFrame
from typing import List

#sklearn
from sklearn.metrics import confusion_matrix


def count_missing(
                target_df:pd.DataFrame, 
                task:str, 
                mask:bool, 
                labels:list
                ) -> np.ndarray:
    
    """
    Creates a Confusion Matrix and fills the rows representing 
    where for a TARGET/TRUE segment there is no PREDICTED segment. 

    Which segments these are are dictated but the match_mask input
    
    IF we have the following confusion matrix, we fill the cells filled with X

    # T\P | A | B | C | NO |
    # ----------------------
    # A   |   |   |   |  X |
    #-----------------------
    # B   |   |   |   |  X |
    #-----------------------
    # C   |   |   |   |  X | 
    #-----------------------
    # NO  |   |   |   |    | 


    These will be the FN

    """
    #we add 1 for NO MATCH
    n_labels = len(labels) + 1

    #empty confusion matrix
    cm = np.zeros((n_labels, n_labels))

    #  To make sure we have an index in our counts vector for all labels
    # we create a zero vector
    zeros = pd.Series(data=[0]*(n_labels-1), index=labels)

    # We get the counts of label for each TARGET segments where there is no matching
    # PREDICTED segment (~match_mask). 
    # EXAMPLE: How many times did we miss to find a matching segment for  a segment with label A?
    counts = (target_df.loc[~mask, task].value_counts() + zeros).fillna(0)

    cm[:-1,-1] = counts[labels].to_numpy()

    return cm


def count_predicted(
                    pred_df: pd.DataFrame,
                    target_df:pd.DataFrame, 
                    task:str, 
                    labels:list, 
                    target_mask: np.ndarray,
                    pred_mask: np.ndarray
                    ) -> np.ndarray:

    """
    Creates a Confusion Matrix and fills the rows representing
    PREDICTED segments

    
    IF we have the following confusion matrix we fill it with two types
    of values: 

    + = Predicted Segments which match a TARGET segment
    - = Predicted Segments which DO NOT match a TARGET segment

    # T\P | A | B | C | NO |
    # ----------------------
    # A   | + | + | + |    |
    #-----------------------
    # B   | + | + | + |    |
    #-----------------------
    # C   | + | + | + |    | 
    #-----------------------
    # NO  | - | - | - |    | 

    """
    #we add 1 for NO MATCH
    n_labels = len(labels) + 1

    #empty confusion matrix
    cm = np.zeros((n_labels, n_labels))

    #  ADDING +
    # First we count the predictions for all predicted segments
    # which DO have a matching TARGET segment
    # we sort both based on j
    matching_targets = target_df.loc[target_mask].sort_values(by="seg_id")[task].to_numpy()
    matching_preds = pred_df.loc[pred_mask].sort_values(by="j")[task].to_numpy()

    if len(matching_targets) and len(matching_preds):
        cm[:-1, :-1] +=  confusion_matrix(
                                            y_true = matching_targets,
                                            y_pred = matching_preds,
                                            labels = list(range(len(labels)))
                                            )


    #  To make sure we have an index in our counts vector for all labels
    # we create a zero vector
    zeros = pd.Series(data=[0] * (n_labels-1), index=labels)

    # ADDING -
    # then we need to to look at all the predicted segments which 
    # DO NOT do not count as having a true match/a target segment match
    counts = (pred_df.loc[~pred_mask, task].value_counts() + zeros).fillna(0)
    cm[-1, :-1] += counts[labels].to_numpy()

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
        scores[f"{prefix}{label}-f1"] = f1

        task_TP += TP
        task_FP += FP
        task_FN += FN
    
    scores[f"{prefix}f1"] = np.mean(list(scores.values()))

    # When the the metric is used in https://arxiv.org/pdf/1704.06104.pdf,
    # they use the micro f1, as they sum the TP and FP and FNs over all labels
    scores[f"{prefix}f1-micro"] = (2*task_TP) / ((2*task_TP) + task_FP + task_FN)

    return scores


def overlap_metric(pred_df:pd.DataFrame, target_df:pd.DataFrame, task_labels:dict, ratios: List[float] = [0.5, 1.0]):

    """
    Metric is from the following paper ( chapt. 6.1 Experimental Setup, p. 1390):

    https://aclanthology.org/N16-1164.pdf
    

    NOTE!
            i = predicted segments (source)
            j = TARGET/ground truth/True segment (source)

            i-t = target of i (in target in source -> target)
            j-t = target of j (in target in source -> target)
    """

    calc_lable_metric = "label" in task_labels
    calc_link_lable_metric = "link_label" in task_labels and len(pred_df["target_id"].unique()) == 0
    

    target_df = target_df.groupby("seg_id", sort = False).first()
    pred_df = pred_df.groupby("seg_id", sort = False).first()

    
    metrics = {}
    metric_stack  = {}
    metric_stack.update({f"f1-{ratio}-micro":[] for ratio in ratios})
    metric_stack.update({f"f1-{ratio}":[] for ratio in ratios})

    for ratio in ratios:

        # TARGET seg that overlap with some PREDICTED seg
        # where j has a i which match over ratio
        source_mask = target_df["i_ratio"].to_numpy() >= ratio

        # PREDICTED seg overlap with some TARGET seg
        # i match a j over a ratio
        p_source_mask = pred_df["j_ratio"].to_numpy() >= ratio

        if calc_lable_metric:

            label_cm = count_missing(
                    target_df = target_df,
                    task = "label", 
                    labels = task_labels["label"],
                    mask = source_mask
                    )


            label_cm += count_predicted(
                            pred_df = pred_df,
                            target_df = target_df,
                            task = "label", 
                            labels = task_labels["label"], 
                            target_mask = source_mask,
                            pred_mask = p_source_mask
                            )


            metrics.update(calc_f1( 
                        label_cm, 
                        labels = task_labels["label"],
                        prefix =f"label-{ratio}-",
                        task = "label",
                        ))

            metric_stack[f"f1-{ratio}"].append(metrics[f"label-{ratio}-f1"])
            metric_stack[f"f1-{ratio}-micro"].append(metrics[f"label-{ratio}-f1-micro"])




        if calc_link_lable_metric:

            # TARGET seg target whih is the same as PREDICTED seg target
            # for each j-t we check the target i-t of all i that is match for j
            i_t = pred_df.loc[target_df["i"].to_numpy(), "target_id"].to_numpy()
            link_mask = target_df["target_id"].to_numpy() == pred_df.loc[i_t, "j"].to_numpy()

            # PREDICTED seg target is matching a TARGET seg
            # if the i-t of j match over the ratio threshold
            target_match = pred_df.loc[i_t, "j_ratio"].to_numpy() >= ratio


            # PREDICTED seg target is the same as TARGET seg target
            # if i-t is the same as j-t
            p_link_mask = pred_df.loc[pred_df["target_id"], "j"].to_numpy()  == target_df.loc[pred_df["j"].to_numpy(), "target_id"].to_numpy()

            # PREDICTED seg target overlap with some TARGET SEG
            # if the segment target/link destination of the predicted segments is overlapping with a target segment
            p_target_match = pred_df.loc[pred_df["target_id"], "j_ratio"].to_numpy() >= ratio


            link_label_cm = count_missing(
                        target_df = target_df,
                        task = "link_label", 
                        labels = task_labels["link_label"],
                        mask = source_mask * link_mask * target_match
                        )


            link_label_cm += count_predicted(
                            pred_df = pred_df,
                            target_df = target_df, 
                            task = "link_label", 
                            labels = task_labels["link_label"], 
                            target_mask = source_mask * link_mask * target_match,
                            pred_mask = p_source_mask * p_link_mask * p_target_match
                            )

            metrics.update(calc_f1( 
                    link_label_cm, 
                    labels = task_labels["link_label"],
                    prefix = f"link_label-{ratio}-",
                    task = "link_label"
                    ))

            metric_stack[f"f1-{ratio}"].append(metrics[f"link_label-{ratio}-f1"])
            metric_stack[f"f1-{ratio}-micro"].append(metrics[f"link_label-{ratio}-f1-micro"])


    for k,v in metric_stack.items():
        metrics[k] = np.mean(v)

    return metrics
