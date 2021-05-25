
#basics
import numpy as np
import pandas as pd


"""
Metric from : https://www.aclweb.org/anthology/N16-1164.pdf

also referensed in https://arxiv.org/pdf/1704.06104.pdf


Given that j is a true argument component and i is an ACC, the formulas for
the ACI task are:

    TP = {j |∃i |gl(j) = pl(i) ∧ i = j} (21)
    FP = {i | pl(i) -= n ∧ -∃j | gl(j) = pl(i) ∧ i = j} (22)
    FN = {j | -∃i |gl(j) = pl(i) ∧ i = j} (23)

where gl(j) is the gold standard label of j, pl(i) is
the predicted label of i, n is the non-argumentative
class, and i = j means i is a match for j.


NOTE! As i = j needs to be true in all the cases it means that if we find segments which is not > 50% match
this is no treated as anything. Its simply ignored

"""


def setup(task_labels:dict):

    # # all FN for a label X will be all cases where true cases of X where
    # # classifed as some other label Y or if the overlap is under 50 or 1, i.e. no unit
    # # was identified. For those cases we set label as "None"
    # if "miss" not in task_labels:
    #     task_labels.append("miss")
    conf_ms = {}
    index_map = {}
    for task, labels in task_labels.items():
        conf_ms[task] = {}
        nr_l = len(labels)
        conf_ms[task]["exact"] = np.zeros((nr_l, nr_l))
        conf_ms[task]["approximate"] = np.zeros((nr_l, nr_l))
        index_map[task] = {l:i for i,e in enumerate(labels)}

    return conf_ms, index_map


def create_confusion_matrixes(df:pd.DataFrame, task_labels:dict):
    
    samples = df.groupby()
    
    # we remove link as we dont caclulate the f1 for this task,
    # its used to calculate the link_label f1 (e.g. refered to as "relation f1" in paper)
    task_labels = deepcopy(task_labels)
    task_labels.pop("link")
    
    conf_ms, index_map = create_empty_conf_ms(task_labels)

    for i,sample in samples:

        target_segments = sample.groupby()
        #predicted_segments = sample.groupby()
        
        link_label_tabel = {
                                "T-link_label": [],
                                "T-link": [],
                                "link_label": [],
                                "link": [],
                                "50%": [],
                                "100%": [],
                                }

        for i, segment in target_segments:

            app_match = False
            exact_match = False
            link = None
            link_label = None
            
            # now we go through all the predicted segments which are inside the ground truth segment. 
            # we first do this over the segment label, and save the overlap information for a second loop
            # for link_label f1
            for j, pred_segment in segment.groupby():
                
                overlap_percent = pred_segment.shape[0] / segment.shape[0]

                link_label = index_map["link_label"][pred_segment["link_label"].unique().tolist()[0]]
                link = index_map["link"][pred_segment["link"].unique().tolist()[0]]

                ti = index_map["label"][segment[f"T-label"].unique().tolist()[0]]
                pi = index_map["label"][pred_segment["label"].unique().tolist()[0]]

                if overlap_percent == 1:
                    conf_ms["label"]["exact"][ti, pi] += 1
                    exact_match = True
    

                if overlap_percent > 0.5:
                    conf_ms["label"]["approximate"][ti, pi] += 1
                    app_match = True


                    # if we find something that overlaps over 50% we can break
                    # the loop, as there will be no other matches.
                    break

            
            link_label_tabel["50%"].append(app_match)
            link_label_tabel["100%"].append(exact_match)
            link_label_tabel["link"].append(link)
            link_label_tabel["link_label"].append(link_label)
            link_label_tabel["T-link_label"].append(segment["T-link_label"].unique().tolist()[0])
            link_label_tabel["T-link"].append(segment["T-link"].unique().tolist()[0])

        link_label_df = pd.DataFrame(link_label_tabel)

        # we do a second loop over the true segments as we relations can go forward or backwards, so we need to know
        # the over lap of future segments
        for i,row in link_label_df.iterrows():

            row["link"]
            row["pred_link"]

            sample_link_label_info.append(link_label_info)
            








                               if task == "link_label":

                        # link_label target and prediction
                        ti = index_map[task][segment[f"T-{task}"].unique().tolist()[0]]
                        pi = index_map[task]pred_segment[task].unique().tolist()[0]]

                        # link target and prediction
                        tli = index_map[task][segment[f"T-{task}"].unique().tolist()[0]]
                        pli = index_map[task]pred_segment[task].unique().tolist()[0]]

            
                    else:
            
            if overlap_found:


                p_idx = set(pred_segment.index.tolist())

                pred_label = pred_segment["preds"].unique().tolist()[0]
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
                    df:pd.DataFrame,
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





