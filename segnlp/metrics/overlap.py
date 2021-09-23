
#basics
import numpy as np
import pandas as pd
from collections import Counter
from pandas.core.frame import DataFrame
from typing import List

#sklearn
from sklearn.metrics import confusion_matrix

# segnlp
from .metric_utils import overlap_ratio


def get_missing(
                target_df:pd.DataFrame, 
                task:str, 
                match_mask:bool, 
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
    counts = (target_df.loc[~match_mask, task].value_counts() + zeros).fillna(0)

    print(counts)
    print(counts[labels].to_numpy())
    cm[:-1,-1] = counts[labels].to_numpy()

    return cm


def count_predicted(
                    pred_df: pd.Dataframe,
                    target_df:pd.DataFrame, 
                    task:str, 
                    labels:list, 
                    match_mask: np.ndarray
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

    # First we count the predictions for all predicted segments
    # which DO have a matching TARGET segment
    cm[:-1, :-1] +=  confusion_matrix(
                                        target_df.loc[match_mask, "label"].to_numpy(),
                                        pred_df.loc[match_mask, "label"].to_numpy(),
                                        labels = labels
                                        )

    #  To make sure we have an index in our counts vector for all labels
    # we create a zero vector
    zeros = pd.Series(data=[0] * (n_labels-1), index=labels)

    # then we need to to look at all the predicted segments which 
    # DO NOT do not count as having a true match/a target segment match
    counts = (pred_df.loc[~match_mask, task].value_counts() + zeros).fillna(0)
    cm[-1, :-1] = counts[labels].to_numpy()


    return cm


def calc_f1(cm:np.array, labels:list, prefix:str):
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


def overlap_metric(pred_df:pd.DataFrame, target_df:pd.DataFrame, task_labels:dict, ratios: List[float]):

    """
    Metric is from the following paper ( chapt. 6.1 Experimental Setup, p. 1390):

    https://aclanthology.org/N16-1164.pdf
    

    NOTE!
            i = predicted segments (source)
            j = TARGET/ground truth/True segment (source)

            i-t = target of i (in target in source -> target)
            j-t = target of j (in target in source -> target)
    """
    
    target_df = target_df.groupby("seg_id", sort = False).first()
    pred_df = pred_df.groupby("seg_id", sort = False).first()


    for ratio in ratios:


        # where j has a i which match over ratio
        source_mask = target_df["match_ratio"] >= ratio

        
        # which j where j-t == i-t, i.e. which target segments which have
        # a matching predicted segment which linked target is matching with link
        # target of the target segment.
        i_t = target_df.loc[target_df["link"], "best_match_i"]
        link_mask = target_df["link"] == pred_df.loc[i_t, "best_match_j"] 

        # if the i-t of j match over the ratio threshold
        target_match = pred_df.loc[i_t, "match_ratio"] >= ratio


        count_missing(
                    target_df = target_df,
                    task = "label", 
                    labels = task_labels["label"],
                    mask = source_mask
                    )

        count_missing(
                    target_df = target_df,
                    task = "link_label", 
                    labels = task_labels["link_label"],
                    mask = np.logical_and(source_mask, link_mask, target_match)
                    )


        # i match a j over ratio
        source_mask = pred_df["match_ratio"] >= ratio

        # if i-t is the same as j-t
        link_mask = pred_df.loc[pred_df["link"], "best_match_j"]  == target_df.loc[pred_df["best_match_j"], "link"]

        # if the segment target/link destination of the predicted segments is overlapping with a target segment
        target_match = pred_df.loc[pred_df["link"], "ratio"] >= ratio


        count_predicted(
                        pred_df = pred_df,
                        target_df = target_df, 
                        task = "label", 
                        labels = task_labels["label"], 
                        mask = 

                        )



        calc_f1( 
                lcms[0], 
                labels = task_labels["label"],
                prefix =f"label-100%-",
                )


        calc_f1( 
                lcms[0], 
                labels = task_labels["label"],
                prefix =f"label-100%-",
                )


    
    # # we remap all i to j, if we dont find any i that maps to a j
    # # we swap j to -1 (map defults to NaN but we change to -1)
    # pred_df["link-j-100%"] = pred_df["link"].map(i2j_exact).fillna(-1).to_numpy()
    # pred_df["link-j-50%"] = pred_df["link"].map(i2j_approx).fillna(-1).to_numpy()

    # # # we create confusion matrixes for each task and for each overlap ratio (100%, 50%)
    # # # get_missing_counts() will fill the part of the cm where predictions are missing.
    # # #only FN, last column which is for "NO_MATCH"
    # # label_cms, link_label_cms = get_missing_counts(target_df, exact, approx, task_labels)
    
    # # # get_pred_counts will fill values in cm that count when predictics are correct and when
    # # # they are predicting on segments which are not true preds. i.e. TP, FP (and FN, as ones TP is anothers FN)
    # # label_cms_p, link_label_cms_p = get_pred_counts(pred_df, target_df, exact, approx, task_labels, i)
    
    # #add them together, (100% is on index 0, 50% on index 1)
    # lcms = label_cms + label_cms_p
    # llcms = link_label_cms + link_label_cms_p

    # #then we calculate the F1s
    # metrics = {}
    # metrics.update(calc_f1( 
    #                     lcms[0], 
    #                     labels = task_labels["label"],
    #                     prefix =f"label-100%-",
    #                     )
    #                 )

    # metrics.update(calc_f1( 
    #                         lcms[1], 
    #                         labels = task_labels["label"],
    #                         prefix = f"label-50%-",
    #                         )
    #                 )

    # metrics.update(calc_f1( 
    #                         llcms[0], 
    #                         labels = task_labels["link_label"],
    #                         prefix = f"link_label-100%-",
    #                         )
    #                 )
    # metrics.update(calc_f1( 
    #                         llcms[1], 
    #                         labels = task_labels["link_label"],
    #                         prefix = f"link_label-50%-",
    #                         )
    #                 )

    
    metrics["f1-50%"] = (metrics["link_label-50%-f1"] + metrics["label-50%-f1"]) / 2
    metrics["f1-100%"] = (metrics["link_label-100%-f1"] + metrics["label-100%-f1"]) / 2

    metrics["f1-50%-micro"] = (metrics["link_label-50%-f1-micro"] + metrics["label-50%-f1-micro"]) / 2
    metrics["f1-100%-micro"] = (metrics["link_label-100%-f1-micro"] + metrics["label-100%-f1-micro"]) / 2
    
    return metrics
