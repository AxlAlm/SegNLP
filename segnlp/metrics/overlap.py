
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



For a label X we TP, FP and FN are defined as:

        a TP is a i which overlaps with j and j and i has label X

        a FP is :
                1) a i which is not overlapping with j, 
                or
                2) a i which overlaps with j where i is labled X and j is Z

        a FN is :
                1) a j which has no i which overlap with it, 
                or
                2) a i which overlaps with j but i is labled Z and j is labeled X


"""

def overlap(
            target_segs:dict, 
            pred_seg:pd.DataFrame, 
            labels:list,
            task: str,
            ):

    match_info = {
                    "exact":None,
                    "approx": None,
                    "j": None,
                    "target_label": None,
                    "pred_label": None,
                }

    t_segs = pred_seg.groupby("T-seg_id")

    if not t_seg_ids:
        return match_info

    for t_seg_id, overlap_df in t_segs:
        target_df = target_segs[t_seg_id]
        overlap_percent = overlap_df.shape / target_df.shape

        match_info["target_label"] =  labels.index(target_df[task].tolist()[0])
        match_info["pred_label"] = labels.index(overlap_df[task].tolist()[0])

        if overlap_percent > 0.99:
            match_info["exact"] = True
            match_info["approx"] = True
            match_info["j"] = t_seg_id
        
        elif overlap_percent > 0.5:
            match_info["approx"] = True
            match_info["j"] = t_seg_id
    
    return match_info


def fill_pred_labels(
                    i, 
                    conf_ms:dict, 
                    target_segs:list, 
                    pred_seg:pd.DataFrame, 
                    j2match_info:dict, 
                    i2j:dict,
                    labels:list,
                    ):

        #if something is an exact match its also going to be an approximate match so we dont need the returned labels etc
        match_info = overlap(
                                target_segs = target_segs,
                                pred_seg = pred_seg,
                                labels = labels,
                                task="label",
                            )

        j2match_info[j] = match_info
        i2j[i] = match_info["j"]

        t = match_info["target_label"]
        p = match_info["pred_label"]

        if approx:
            conf_ms["label"]["approx"][t, p] += 1
        else:
            conf_ms["label"]["approx"][-1, p] += 1

        if exact:
            conf_ms["label"]["exact"][t, p] += 1
        else:
            conf_ms["label"]["exact"][-1, p] += 1


def fill_pred_link_labels(  i, 
                            conf_ms:dict, 
                            target_segs:list, 
                            pred_seg:pd.DataFrame, 
                            i2j:dict, 
                            j2match_info:dict,
                            labels:list,
                            ):

    source = pred_seg
    link = pred_seg["link"].tolist()[0]
    target = pred_segs[link]

    #if something is an exact match its also going to be an approximate match so we dont need the returned labels etc
    source_match_info = overlap(
                                    target_segs = target_segs,
                                    pred_seg = source,
                                    labels = labels,
                                    task = "link_label"

                                )

    target_match_info = overlap(
                                    target_segs = target_segs,
                                    pred_seg = source,
                                    labels = labels,
                                    task = "link_label"
                                )

    t = source_match_info["target_label"]
    p = source_match_info["pred_label"]
    tj = target_match_info["j"]

    j2match_info["link_info"] = {"exact":False, "approx":True}
           
    for overlap in ["approx", "exact"]:

        # check if source and target both overlap with ground truth segments
        #  if they do not, we treat the case as a FP, i.e. set the True label to NO OVERLAP
        if source_match_info[overlap] and target_match_info[overlap]:

            # then we check if the target is the ground truth target
            # i.e. is the predicted link the same as the ground truth link
            # IF its not we treat it as a FP, i.e. set the True label to NO OVERLAP
            if i2j.get(link, "") == tj:
                conf_ms["link_label"][overlap][t, p] += 1
                j2match_info["link_info"][overlap] = True
            else:
                conf_ms["link_label"][overlap][-1, p] += 1
        
        else:
            conf_ms["link_label"][overlap][-1, p] += 1


def fill_missing_label(j, t, j2match_info:dict):

    # if there is not any predicted segments that overlaps with j we count it
    # as NO OVERLAP
    if j not in j2match_info:
        conf_ms["label"]["exact"][t, -1] += 1
        conf_ms["label"]["approx"][t, -1] += 1
    else:
        
        # if we have an overlap but its not exact or approx, we treat it as NO OVERLAP
        if not j2match_info[j]["exact"]:
            conf_ms["label"]["exact"][t, -1] += 1

        if not j2match_info[j]["approx"]:
            conf_ms["label"]["approx"][t, -1] += 1
        

def fill_missing_link_label(j, t, j2match_info:dict):

    # if there is not any predicted segments that overlaps with j we count it
    # as NO OVERLAP
    if j not in j2match_info:
        conf_ms["label"]["exact"][t, -1] += 1
        conf_ms["label"]["approx"][t, -1] += 1
    else:

        # if we have an overlap but there the link is wrong or source or target is not overlapping with
        # correct ground truth segment we treat it as NO OVERLAP
        link_info = j2match_info[j]["link_info"]
        
        if not link_info["exact"]:
            conf_ms["label"]["exact"][t, -1] += 1

        if not link_info["approx"]::
            conf_ms["label"]["approx"][t, -1] += 1


def create_cms(df:pd.DataFrame, task_labels:dict):
    
    conf_ms = {}
    for task, labels in task_labels.items():
        conf_ms[task] = {}
        nr_l = len(labels)
        conf_ms[task]["exact"] = np.zeros((nr_l, nr_l))
        conf_ms[task]["approximate"] = np.zeros((nr_l, nr_l))


    for _, sample in df.groupby("sample_id"):

        target_segs = sample.groupby("T-seg_id")
        pred_segs = sample.groupby("seg_id")

        j2i = {}
        j2match_info = {}

        for i, pred_seg in pred_segs:
            
            fill_pred_labels(
                            i,
                            conf_ms = conf_ms,
                            target_segs = target_segs, 
                            pred_seg = pred_seg,
                            j2match_info = j2match_info,
                            j2i = j2i,
                            labels = task_labels["label"]
                            )

            fill_pred_link_labels(
                                    i, 
                                    conf_ms = conf_ms,
                                    target_segs = target_segs, 
                                    pred_seg = pred_seg,
                                    j2i = j2i,
                                    labels = task_labels["label"]
                                    )


        for j, target_seg in target_segs:
            t = target_seg["T-link_label"].tolist()[0]
            t = task_labels["link_label"].index(t)

            fill_missing_label(
                                j, 
                                t, 
                                j2match_info=j2match_info
                                )

            fill_missing_link_label(
                                    j, 
                                    t, 
                                    j2match_info=j2match_info
                                    )


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

    for i,label in enumerate(task_labels):
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


def overlap_metric(
                    df:pd.DataFrame,
                    task_labels:list,
                    ):

    """

    We create the following confusion matrixes for each task and for EXACT
    and APPROXIMATE matching.

    A, B anc C are just example labels while "NO" stands for "NO OVERLAP",
    which row or column we fill when we have a predicted segments which is not
    overlapping with a ground truth segments or where there is no predicted segments that
    overlaps with a ground truth segment.

    # T\P | A | B | C | NO |
    # ----------------------
    # A   |   |   |   |    |
    #-----------------------
    # B   |   |   |   |    |
    #-----------------------
    # C   |   |   |   |    | 
    #-----------------------
    # NO  |   |   |   |    |  

    """

    conf_ms = create_cms(
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







# def fill_label_cm(sample:pd.DataFrame, conf_ms:dict, index_map:dict):

#     target_segments = sample.groupby()

#     # this dict will contain a mapping between predicted segment ids and the ground truth segments id
#     # that they represent, i.e. overlap with.
#     j2i = {}

#     # we save info for each ground truth segments about which predicted segments is overlapping with it etc
#     overlap_info = []

#     i = 0
#     for j, segment in target_segments:
        
#         ti = index_map["label"][segment[f"T-label"].unique().tolist()[0]]

#         approx_match = False
#         exact_match = False
#         link = None
#         link_label = None
        
#         # now we go through all the predicted segments which are inside the ground truth segment. 
#         # we first do this over the segment label, and save the overlap information for a second loop
#         # for link_label f1
#         for _, pred_segment in segment.groupby():
#             #match_found = False
            
#             overlap_percent = pred_segment.shape[0] / segment.shape[0]

#             pi = index_map["label"][pred_segment["label"].unique().tolist()[0]]

#             if overlap_percent == 1:
#                 conf_ms["label"]["exact"][ti, pi] += 1
#                 exact_match = True
#             else:
#                 # We count a segments which is not overlapping
#                 # as FP. We set the true label as no overlap
#                 conf_ms["label"]["exact"][-1, pi] += 1


#             if overlap_percent > 0.5:
#                 conf_ms["label"]["approximate"][ti, pi] += 1
#                 approx_match = True

#                 link_label = index_map["link_label"][pred_segment["link_label"].unique().tolist()[0]]
#                 link = index_map["link"][pred_segment["link"].unique().tolist()[0]]
                
#                 i2j[i] = j
    
#             else:
#                 # We count a segments which is not overlapping
#                 # as FP. We set the true label as no overlap                    
#                 conf_ms["label"]["approximate"][-1, pi] += 1

#             i += 1


#         # if we are missing any overlapping segments we treat is as a FN for label ti
#         if not exact_match:
#             conf_ms["label"]["exact"][t1, -1] += 1

#         if not approx_match:
#             conf_ms["label"]["approximate"][t1, -1] += 1


#         overlap_info.append({    
#                             "T-link_label": segment["T-link_label"].unique().tolist()[0],
#                             "T-link": segment["T-link"].unique().tolist()[0],
#                             "link_label": link_label,
#                             "link": link,
#                             "exact": exact_match,
#                             "approximate": approx_match,
#                             })
    
#     return overlap_info, j2i



# def fill_link_label_cm(overlap_info:list, i2j:dict, conf_ms:dict, index_map:dict):


#     target_segment_info

#     for i, source in enumerate(predicted_segment_info):
#         target = predicted_segment_info[source["link"]]


#         # check if soure and taget both overlap with ground truth segments
#         #  if they do not, we treat the case as a FP, i.e. set the True label to NO OVERLAP
#         if source["exact"] and target["exact"]:

#             # then we check if the target is the ground truth target
#             # i.e. is the predicted link the same as the ground truth link
#             # IF its not we treat it as a FP, i.e. set the True label to NO OVERLAP
#             if i2j.get(source["link"]) == ground_truth["link"]:
#                 conf_ms["link_label"]["exact"][ti, pi] += 1
#             else:
#                 conf_ms["link_label"]["exact"][-1, pi] += 1
        
#         else:
#             conf_ms["link_label"]["exact"][-1, pi] += 1








#         # if either source or target is not an approximate match
#         # we treat it as a FP
#         if source["approximate"] and target["approximate"]:


#         else
#             conf_ms["link_label"]["approximate"][-1, pi] += 1



#         if 


#         if not source["exact"]:
#             source_exact = True

#         if not seg_info["approximate"]:





#             conf_ms["link_label"]["exact"][-1, pi] += 1
#                     conf_ms["link_label"]["exact"][-1, pi] += 1


#         if predicted_segment_info[seg_info["link"]]


#     # we do a second loop over the true segments as we relations can go forward or backwards, so we need to know
#     # the over lap of future segments
#     for dict_ in seg_results:
        
#         #if the source is not overlapping with anything
#         if not dict_["exact"]:


#         ij = i2j.get(dict_["link"], None)

#         # if we cannot find any
#         if ij is None:

        
#         ti = index_map["link_label"][dict_["T-link_label"]]
#         pi = index_map["link_label"][dict_["link_label"]]


#         # in previous steps we mapped predicted segments j with the overlapping ground truth segments i
#         # i is ith segments of all ground truth segments
#         # j is jth segments of all ground truth segments
#         # j2i(j) is the i of the ground truth segment that j is overlapping with.
#         #
#         # if j2i(j) != i where i and j are predicted link indexes, we 
#         # know that the predicted link is wrong.
#         if j2i.get(dict_["link"], None) != dict_["T-link"]:
#             continue
        



#         if dict_["exact"]:
#             conf_ms["link_label"]["exact"][ti, pi] += 1

#         if dict_["approximate"]:
#             conf_ms["link_label"]["approximate"][ti, pi] += 1



# def create_confusion_matrixes(df:pd.DataFrame, task_labels:dict):
    
#     samples = df.groupby()
    
#     # we remove link as we dont caclulate the f1 for this task,
#     # its used to calculate the link_label f1 (e.g. refered to as "relation f1" in paper)
#     task_labels = deepcopy(task_labels)
#     task_labels.pop("link")
    
#     conf_ms, index_map = create_empty_conf_ms(task_labels)

#     for _,sample in samples:

#         # first we take care of label results. We fill the label confusion matrix
#         # while also extract information about which predicted segments are overlapping with 
#         # which ground truth segments. And create a mapping from predicted segment ids to ground 
#         # truth ids
#         overlap_info, i2j = fill_label_cm(
#                                         sample=sample,
#                                         conf_ms=conf_ms,
#                                         index_map=index_map
#                                         )
    
#         fill_link_label_cm(
#                             overlap_info = overlap_info,
#                             i2j=i2j,
#                             conf_ms = conf_ms,
#                             index_map = index_map
#                             )
  
#     return conf_ms

