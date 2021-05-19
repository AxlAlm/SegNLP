
#basics
import numpy as np
import pandas as pd


"""
Metric from : https://www.aclweb.org/anthology/N16-1164.pdf

also referensed in https://arxiv.org/pdf/1704.06104.pdf

"""


def calc_link_label_matrix(                
                            targets:np.ndarray, 
                            preds:np.ndarray, 
                            target_unit_id:np.ndarray,
                            preds_unit_id:np.ndarray,
                            task_labels:list,
                            ):

    task_labels = task_labels.copy()

    # all FN for a label X will be all cases where true cases of X where
    # classifed as some other label Y or if the overlap is under 50 or 1, i.e. no unit
    # was identified. For those cases we set label as "None"
    if "miss" not in task_labels:
        task_labels.append("miss")

    label_idx = lambda x: task_labels.index(x)

    exact_cm = np.zeros((task_labels-1, task_labels))
    approximate_cm = np.zeros((task_labels-1, task_labels))

    mli = label_idx("miss")

    for i in range(len(samples)):

        df = pd.DataFrame({
                            "targets":targets[i], 
                            "preds":preds[i],
                            "target_unit_id":target_unit_id[i],
                            "preds_unit_id":preds_unit_id[i],
                            })

        units = [g for g in df.groupby("target_unit_id")]

        for tid, tdf in units:
            t_idx = set(tdf.index.tolist())
            tsize = len(t_idx)
            target_label = tdf["targets"].unique().tolist()[0]
            tli = label_idx(target_label)

            no_overlap = True

            for pid, pdf in tdf.groupby("preds_unit_id"):
                p_idx = set(pdf.index.tolist())
                pred_label = pdf["preds"].unique().tolist()[0]
                pli = label_idx(pred_label)

                #calculates the portion of the predicted span overlaps with target span
                overlap_percent = set(t_idx).union(p_idx) / tsize

                if overlap_percent >= 0.5:
                    approximate_cm[tli, pli] += 1
                    no_overlap = False

                if overlap_percent == 1:
                    exact_cm[tli, pli] += 1
                
            if no_overlap:   
                exact_cm[tli, mli] += 1
                approximate_cm[tli, mli] += 1

    return exact_cm, approximate_cm


def calc_confusion_matrix(                
                            targets:np.ndarray, 
                            preds:np.ndarray, 
                            target_unit_id:np.ndarray,
                            preds_unit_id:np.ndarray,
                            task_labels:list,
                            ):

    task_labels = task_labels.copy()

    # all FN for a label X will be all cases where true cases of X where
    # classifed as some other label Y or if the overlap is under 50 or 1, i.e. no unit
    # was identified. For those cases we set label as "None"
    if "miss" not in task_labels:
        task_labels.append("miss")

    label_idx = lambda x: task_labels.index(x)

    exact_cm = np.zeros((task_labels-1, task_labels))
    approximate_cm = np.zeros((task_labels-1, task_labels))

    mli = label_idx("miss")

    for i in range(len(samples)):

        df = pd.DataFrame({
                            "targets":targets[i], 
                            "preds":preds[i],
                            "target_unit_id":target_unit_id[i],
                            "preds_unit_id":preds_unit_id[i],
                            })

        units = [g for g in df.groupby("target_unit_id")]

        for tid, tdf in units:
            t_idx = set(tdf.index.tolist())
            tsize = len(t_idx)
            target_label = tdf["targets"].unique().tolist()[0]
            tli = label_idx(target_label)

            no_overlap = True

            for pid, pdf in tdf.groupby("preds_unit_id"):
                p_idx = set(pdf.index.tolist())
                pred_label = pdf["preds"].unique().tolist()[0]
                pli = label_idx(pred_label)

                #calculates the portion of the predicted span overlaps with target span
                overlap_percent = set(t_idx).union(p_idx) / tsize

                if overlap_percent >= 0.5:
                    approximate_cm[tli, pli] += 1
                    no_overlap = False

                if overlap_percent == 1:
                    exact_cm[tli, pli] += 1
                
            if no_overlap:   
                exact_cm[tli, mli] += 1
                approximate_cm[tli, mli] += 1

    return exact_cm, approximate_cm


def calc_f1(cm, task_labels):
    """

    If we are calculating the F1 for a label the we first calculate the
    TP, FP and FN in the follwing way using the confuson matrix we created.


    For label A:

    # T\P | A | B | C | miss|
    # -----------------------
    # A   | TP| FN| FN| FN  |
    #------------------------
    # B   | FP|   |   |     | 
    #------------------------
    # C   | FP|   |   |     |


    For label B

    # T\P | A | B | C | miss|
    # -----------------------
    # A   |   | FP|   |     |
    #------------------------
    # B   | FN| TP| FN|  FN | 
    #------------------------
    # C   |   | FP|   |     |


    then we se the following formula fot the the f1
    
    f1 = (2*TP) / ((2*TP) + FP + FN)


    We then average the f1 across the labels to get the f1-macro
    """

    scores = {}
    for i,label in enumerate(task_labels):
        TP = cm[i,i]
        
        #all predictions of label which where actually other labels (minus "miss")
        FP = sum(cm[,i]) - TP

        # total miss + 
        FN = sum(cm[i]) - TP
    
        f1 = (2*TP) / ((2*TP) + FP + FN)
        scores[f"{label}-f1"] = f1
    
    scores["f1"] = np.mean(list(scores.values()))
    return scores


def overlap_link_label_metric(
                            target_link_label:np.ndarray, 
                            target_link:np.ndarray,
                            link_label_preds:np.ndarray, 
                            link_preds:np.ndarray, 
                            target_unit_id:np.ndarray,
                            preds_unit_id:np.ndarray,
                            task:str,
                            task_labels:list,
                            ):

    """
    same principle as overlap_metric, but to calculate the metric for 

    """
    exact_cm, approximate_cm = calc_confusion_matrix(
                                                    targets=targets, 
                                                    preds=preds, 
                                                    target_unit_id=target_unit_id,
                                                    preds_unit_id=preds_unit_id,
                                                    task_labels=task_labels,
                                                    )

    exact_scores = calc_f1(exact_cm, task_labels)
    exact_scores = {f"{task}-100%-{k}":v for k,v in exact_scores.items()}

    approximate_scores = calc_f1(approximate_cm, task_labels)
    approximate_scores = {f"{task}-50%-{k}":v for k,v in approximate_scores.items()}
    
    return {**exact_scores, **approximate_scores}


def overlap_metric(
                targets:np.ndarray, 
                preds:np.ndarray, 
                target_unit_id:np.ndarray,
                preds_unit_id:np.ndarray,
                task:str,
                task_labels:list,
                ):

    # We then create two confusion matrix which looks like this 
    # where  A B and C are example labels and miss is when there is no 
    # identifed segments

    # T\P | A | B | C | miss|
    # -----------------------
    # A   |   |   |   |     |
    #------------------------
    # B   |   |   |   |     |
    #------------------------
    # C   |   |   |   |     |
    exact_cm, approximate_cm = calc_confusion_matrix(
                                                    targets=targets, 
                                                    preds=preds, 
                                                    target_unit_id=target_unit_id,
                                                    preds_unit_id=preds_unit_id,
                                                    task_labels=task_labels,
                                                    )

    exact_scores = calc_f1(exact_cm, task_labels)
    exact_scores = {f"{task}-100%-{k}":v for k,v in exact_scores.items()}

    approximate_scores = calc_f1(approximate_cm, task_labels)
    approximate_scores = {f"{task}-50%-{k}":v for k,v in approximate_scores.items()}
    
    return {**exact_scores, **approximate_scores}





