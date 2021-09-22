

#basics
from typing import Tuple
from collections import Counter
import numpy as np
import pandas as pd

"""

Finds the overlap in % of tokens betwen predicted and target segments.

For each target it will:

1) find the predicted segments ids within the target

2) take the largest of these segments, as it will be the best matching

3) check to what degree the token betweens the predicted segment and and target segment overlap

"""

def _overlap(target_seg_df: pd.DataFrame, pdf: pd.DataFrame):

    # target segment id
    j = target_seg_df["seg_id"].to_list()[0]

    # predicted segments in the sample target df
    pred_seg_ids = target_seg_df["PRED-seg_id"].dropna().to_list()


    if not pred_seg_ids:
        return np.array([None, None, None])
    
    #best pred segment id
    i = Counter(pred_seg_ids).most_common(1)[0][0]

    #get token indexes
    ps_token_ids = set(pdf.loc[[i], "token_id"]) #slowest part
    ts_token_ids = set(target_seg_df["token_id"])

    #calculate the overlap
    overlap_ratio = len(ts_token_ids.intersection(ps_token_ids)) / max(len(ps_token_ids), len(ts_token_ids))

    return np.array([i, j, overlap_ratio])


def overlap_ratio(pred_df : pd.DataFrame, target_df : pd.DataFrame) -> Tuple[np.ndarray]:

    # Create a new dfs which contain the information we need
    pdf = pd.DataFrame({
                        "token_id": pred_df["id"].to_numpy()
                        }, 
                        index = pred_df["seg_id"].to_numpy(),
                        )
    pdf = pdf[~pdf.index.isna()]

    tdf = pd.DataFrame({
                        "token_id": target_df["id"].to_numpy(),
                        "seg_id": target_df["seg_id"].to_numpy(),
                        "PRED-seg_id": pred_df["seg_id"].to_numpy(),
                        }, 
                        index = target_df["seg_id"].to_numpy(),
                        )
    tdf = tdf[~tdf.index.isna()]

    # we extract matching information. Which predicted segments are overlapping with which 
    # ground truth segments
    overlap_info = np.vstack(tdf.groupby(level = 0, sort=False).apply(_overlap, (pdf)))
    
    #seperate the info
    i = overlap_info[:,0].astype(int) #predicted segment id
    j = overlap_info[:,1].astype(int) # ground truth segment id
    ratio = overlap_info[:,2]

    return i, j, ratio