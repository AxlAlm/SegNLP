
#basics
import numpy as np
import pandas as pd
from collections import Counter

#sklearn
from sklearn.metrics import confusion_matrix


def extract_match_info(df):
    

    def overlap(target, preds):

        # target segment id
        j = target["seg_id"].to_list()[0]

        pred_seg_ids = preds["seg_id"].dropna().to_list()

        if not pred_seg_ids:
            return [0, 0, -1, -1]
        
        #best pred segment id
        i = Counter(pred_seg_ids).most_common(1)[0][0]

        #get token indexes
        pred_seg_token_idx = set(preds.loc[[i], "index"]) #slowest part
        target_seg_token_idx = set(target.index)

        #calculate the overlap
        overlap = len(target_seg_token_idx.intersection(pred_seg_token_idx)) / max(len(pred_seg_token_idx), len(target_seg_token_idx))

        approx = 1 if overlap > 0.5 else 0
        exact =  1 if overlap > 0.99 else 0

        return [exact, approx, i if approx else -1, j if approx else -1]


    # create pdf with predicted segments ids as index to be able
    # to select rows faster
    pdf = pd.DataFrame({
                        "seg_id": df.loc["PRED", "seg_id"].to_numpy(),
                        "index": df.loc["PRED"].index.to_numpy(),
                        }, 
                        index = df.loc["PRED", "seg_id"].to_numpy(),
                        )

    # we extract matching information. Which predicted segments are overlapping with which 
    # ground truth segments
    match_info = np.vstack(df.loc["TARGET"].groupby("seg_id", sort=False).apply(overlap, (pdf)))

    exact = match_info[:,0].astype(bool)
    approx = match_info[:,1].astype(bool)
    i = match_info[:,2] #predicted segment id
    j = match_info[:,3] # ground truth segment id
    
    #contains mapping between i j where i is an exact/approx match for j
    i2j_exact = dict(zip(i[exact],j[exact]))
    i2j_approx = dict(zip(i[approx],j[approx]))

    return exact, approx, i2j_exact, i2j_approx, i


def get_missing_counts(df, exact, approx, task_labels):
    target_df = df.loc["TARGET"].groupby("seg_id", sort = False).first()

    #we add 1 for NO MATCH
    n_labels = len(task_labels["label"])+1
    n_link_labels = len(task_labels["link_label"])+1

    label_cms = []
    link_label_cms = []
    for k, bool_mask in zip(["100%", "50%"],[exact, approx]):

        # amount of FN, i.e. how many of each label which are not
        # exact/approx matches 
        
        #empty confusion matrix
        lcm = np.zeros((n_labels, n_labels))

       # create zero counts for all labels and add them to the counts we find. we do this to 
        # make sure we have counts for all labels (make it easier to add to lcm)
        zeros = pd.Series(data=[0]*(n_labels-1), index=task_labels["label"])
        lc = (target_df.loc[~bool_mask,"label"].value_counts() + zeros).fillna(0)

        lcm[:-1,-1] = lc[task_labels["label"]].to_numpy()
        label_cms.append(lcm)


        # We check if 1) source matches then 2) if j-link-PORTION match with target link (j's).
        # if 2) is true it means both that the predicted link is pointing to the correct 
        # target segment and that the segment is a approx/exact match.
        cond1 =  bool_mask
        cond2 = target_df[f"link-j-{k}"] == target_df["link"]
        cond = cond1 & cond2

        #empty confusion matrix
        llcm = np.zeros((n_link_labels, n_link_labels))

        # create zero counts for all labels and add them to the counts we find. we do this to 
        # make sure we have counts for all labels (make it easier to add to llcm)
        zeros = pd.Series(data=[0]*(n_link_labels-1), index = task_labels["link_label"])
        llc = (target_df.loc[~cond, "link_label"].value_counts() + zeros).fillna(0)

        llcm[:-1,-1] = llc[task_labels["link_label"]].to_numpy()

        link_label_cms.append(llcm)


    return np.array(label_cms), np.array(link_label_cms)


def get_pred_counts(df, exact, approx, task_labels, i):

    pred_df =  df.loc["PRED"].groupby("seg_id", sort = False).first()
    target_df =  df.loc["PRED"].groupby("seg_id", sort = False).first()

    #we add 1 for NO MATCH
    n_labels = len(task_labels["label"])+1
    n_link_labels = len(task_labels["link_label"])+1

    label_cms = []
    link_label_cms = []
    for k, bool_mask in zip(["100%", "50%"],[exact, approx]):

        i_list = i[bool_mask]
        cond1 = pred_df.index.isin(i_list)

        lcm = np.zeros((n_labels, n_labels))

        #will create confusion matrix for all segments which are matches.
        try:
            lcm[:-1, :-1] +=  confusion_matrix(
                                target_df.loc[cond1, "label"].to_numpy(),
                                pred_df.loc[cond1, "label"].to_numpy(),
                                labels = task_labels["label"]
                                )
        except ValueError as e:
            pass


        # count labels for all segments which are not matches
        # this will be added to conf matrix on i-axis of label NO_MATCH
        # we add zeros just to make sure we have counts for everything
        zeros = pd.Series(data=[0]*(n_labels-1), index=task_labels["label"])
        lc = (pred_df.loc[~cond1,"label"].value_counts() + zeros).fillna(0)
        lcm[-1, :-1] = lc[task_labels["label"]].to_numpy()

        label_cms.append(lcm)

        llcm = np.zeros((n_link_labels, n_link_labels))

        # # will create a confusion matrix for all link labels which 
        # # match and the relation is true
        cond2 = pred_df[f"link-j-{k}"] == target_df["link"]
        cond_1_2 = cond1 & cond2

        try:
            llcm[:-1, :-1] = confusion_matrix(
                                target_df.loc[cond_1_2,"link_label"].to_numpy(),
                                pred_df.loc[cond_1_2,"link_label"].to_numpy(),
                                labels = task_labels["link_label"]
                                )
        except ValueError as e:
            pass

        # will count labels for all link labels where either link is wrong
        # or source or target is not matching with any ground truth segments.
        # this will be added to conf matrix on i-axis of label NONE 
        #  we add zeros to make sure we have the counts for all labels      
        zeros = pd.Series(data=[0]*(n_link_labels-1), index=task_labels["link_label"])
        llc = (pred_df.loc[~cond_1_2,"link_label"].value_counts() + zeros).fillna(0)
        llcm[-1,:-1] = llc[task_labels["link_label"]].to_numpy()

        link_label_cms.append(llcm)

    return np.array(label_cms), np.array(link_label_cms)


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


def overlap_metric(df:pd.DataFrame, task_labels:dict):

    exact, approx, i2j_exact, i2j_approx, i = extract_match_info(df)
    
    # we remap all i to j, if we dont find any i that maps to a j
    # we swap j to -1 (map defults to NaN but we change to -1)
    df.loc["PRED", "link-j-100%"] = df.loc["PRED", "link"].map(i2j_exact).fillna(-1).to_numpy()
    df.loc["PRED", "link-j-50%"] = df.loc["PRED", "link"].map(i2j_approx).fillna(-1).to_numpy()

    # we create confusion matrixes for each task and for each overlap ratio (100%, 50%)
    # get_missing_counts() will fill the part of the cm where predictions are missing.
    #only FN, last column which is for "NO_MATCH"
    label_cms, link_label_cms = get_missing_counts(df, exact, approx, task_labels)
    
    # get_pred_counts will fill values in cm that count when predictics are correct and when
    # they are predicting on segments which are not true preds. i.e. TP, FP (and FN, as ones TP is anothers FN)
    label_cms_p, link_label_cms_p = get_pred_counts(df, exact, approx, task_labels, i)
    
    #add them together, (100% is on index 0, 50% on index 1)
    lcms = label_cms + label_cms_p
    llcms = link_label_cms + link_label_cms_p

    #then we calculate the F1s
    metrics = {}
    metrics.update(calc_f1( 
                        lcms[0], 
                        labels = task_labels["label"],
                        prefix =f"label-100%-",
                        )
                    )

    metrics.update(calc_f1( 
                            lcms[1], 
                            labels = task_labels["label"],
                            prefix = f"label-50%-",
                            )
                    )

    metrics.update(calc_f1( 
                            llcms[0], 
                            labels = task_labels["link_label"],
                            prefix = f"link_label-100%-",
                            )
                    )
    metrics.update(calc_f1( 
                            llcms[1], 
                            labels = task_labels["link_label"],
                            prefix = f"link_label-50%-",
                            )
                    )

    
    metrics["f1-50%"] = metrics["link_label-50%-f1"] + metrics["label-50%-f1"] / 2
    metrics["f1-100%"] = metrics["link_label-100%-f1"] + metrics["label-100%-f1"] / 2

    metrics["f1-50%-micro"] = metrics["link_label-50%-f1-micro"] + metrics["label-50%-f1-micro"] / 2
    metrics["f1-100%-micro"] = metrics["link_label-100%-f1-micro"] + metrics["label-100%-f1-micro"] / 2
    
    return metrics
