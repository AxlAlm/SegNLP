
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
        


        # this dict will contain a mapping between predicted segment ids and the ground truth segments id
        # that they represent, i.e. overlap with.
        j2i = {}

        # segment results
        seg_results = []


        j = 0
        for i, segment in target_segments:

            app_match = False
            exact_match = False
            link = None
            link_label = None
            
            # now we go through all the predicted segments which are inside the ground truth segment. 
            # we first do this over the segment label, and save the overlap information for a second loop
            # for link_label f1
            for _, pred_segment in segment.groupby():
                
                overlap_percent = pred_segment.shape[0] / segment.shape[0]

                ti = index_map["label"][segment[f"T-label"].unique().tolist()[0]]
                pi = index_map["label"][pred_segment["label"].unique().tolist()[0]]

                if overlap_percent == 1:
                    conf_ms["label"]["exact"][ti, pi] += 1
                    exact_match = True
    

                if overlap_percent > 0.5:
                    conf_ms["label"]["approximate"][ti, pi] += 1
                    app_match = True

                    link_label = index_map["link_label"][pred_segment["link_label"].unique().tolist()[0]]
                    link = index_map["link"][pred_segment["link"].unique().tolist()[0]]
                    
                    j2i[j] = i

                    # if we find something that overlaps over 50% we can break
                    # the loop, as there will be no other matches.
                    #break
                
                j += 1


            seg_results.append({    
                                "T-link_label": segment["T-link_label"].unique().tolist()[0],
                                "T-link": segment["T-link"].unique().tolist()[0],
                                "link_label": link_label,
                                "link": link,
                                "50%": app_match,
                                "100%": exact_match,
                                })

        # we do a second loop over the true segments as we relations can go forward or backwards, so we need to know
        # the over lap of future segments
        for dict_ in seg_results:
            
            # in previous steps we mapped predicted segments j with the overlapping ground truth segments i
            # i is ith segments of all ground truth segments
            # j is jth segments of all ground truth segments
            # j2i(j) is the i of the ground truth segment that j is overlapping with.
            #
            # if j2i(j) != i where i and j are predicted link indexes, we 
            # know that the predicted link is wrong.
            if j2i.get(dict_["link"], None) != dict_["T-link"]:
                continue
            

            ti = index_map["link_label"][dict_["T-link_label"]]
            pi = index_map["link_label"][dict_["link_label"]]


            if dict_["100%"]:
                conf_ms["link_label"]["exact"][ti, pi] += 1

            if dict_["50%"]:
                conf_ms["link_label"]["approximate"][ti, pi] += 1


    return conf_ms


def calc_f1(cm:np.array, labels:list, prefix:str):
    """

    If we are calculating the F1 for a label the we first calculate the
    TP, FP and FN in the follwing way using the confuson matrix we created.


    For label A:

    # T\P | A | B | C |
    # -----------------
    # A   | TP| FN| FN|
    #------------------
    # B   | FP|   |   |
    #------------------
    # C   | FP|   |   | 


    For label B

    # T\P | A | B | C |
    # -----------------
    # A   |   | FP|   |
    #------------------
    # B   | FN| TP| FN|  
    #------------------
    # C   |   | FP|   | 


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
        scores[f"{prefix}{label}-f1"] = f1
    
    scores[f"{prefix}f1"] = np.mean(list(scores.values()))
    return scores


def overlap_metric(
                    df:pd.DataFrame,
                    task_labels:list,
                    ):

    # We then create confusion matries which looks like this 
    # where  A B and C are example labels and miss is when there is no 
    # identifed segments

    # T\P | A | B | C |
    # -----------------
    # A   |   |   |   |
    #------------------
    # B   |   |   |   | 
    #------------------
    # C   |   |   |   | 
    conf_ms = create_confusion_matrixes(
                                        df=df, 
                                        task_labels=task_labels
                                        )

    scores = {}
    for task, m_dict in conf_ms:
        scores.update(
                        calc_f1(
                                cm=m_dict["exact"],
                                labels=task_labels[task],
                                prefix=f"{task}-100%-",
                                )
                        )

        scores.update(
                        calc_f1(
                                cm=m_dict["approximate"],
                                labels=task_labels[task],
                                prefix=f"{task}-50%-",
                                )
                    )

    return scores





