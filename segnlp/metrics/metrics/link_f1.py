
#basics
import numpy as np

# sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#segnlp
from segnlp import utils


def link_f1(targets:list, preds:list):

    """
    all links are represented by a number which indicates the index of the related unit.

    we transform each label into a one hot vector, i.e. create adjencency matrixes for each sample. 
    Then we flatten the one hot vectors for each unit in each sample and add it to a large batch array.

    If we have pred link label 2 and we have 5 units in the sample, we create
    the following vector:

        p = [0,0,1,0,0]

    
    if our ground truth is 1 we have 

        t = [0,1,0,0,0]

    score is then caluclated by comparing t and p, which means we have that our metric
    is not just based on the predicted link index (1 vs 2) but also on what it does not predict
    to be a related pair. 

    """

    flat_targets = []
    flat_preds = []
    for st, sp in zip(targets, preds):
        target_adj_m = one_hots(utils.ensure_numpy(st))
        pred_adj_m = one_hots(utils.ensure_numpy(sp))

        flat_targets.extend(target_adj_m.flatten().tolist())
        flat_preds.extend(pred_adj_m.flatten().tolist())

    No_Link_p, Link_p = precision_score(flat_targets, flat_preds, labels=[0,1], average=None)
    No_Link_r, Link_r = recall_score(flat_targets, flat_preds, labels=[0,1], average=None)

    Link_f1 = 2 * ((Link_p * Link_r) / (Link_p + Link_r))
    No_Link_fi = 2 * ((No_Link_p * No_Link_r) / (No_Link_p + No_Link_r))

    link_f1 = (Link_f1 + No_Link_fi) / 2

    return {
            "link-f1":link_f1,
            "LINK-precision":Link_p,
            "LINK-recall":Link_r,
            "LINK-f1":Link_f1,
            "NO-LINK-precision":No_Link_p,
            "NO-LINK-recall":No_Link_r,
            "NO-LINK-f1":No_Link_fi,
            }



# def token_link_f1(targets:np.ndarray, preds:np.ndarray, lengths):

#     """
#     Same as link_metric() but we have extended the metric to apply to token level. We
#     need this as if we are predicting links on predicted segments we cannot align predicted 
#     segments with ground truth segments, we can only align token ground truths with
#     with token predictions.

#     So, the proposed metrics is to simply apply the calculation mention in link_metric() 
#     on tokens instead of units.

#     If we predict link 1 for a unit we remake the prediction to 

#     """

#     flat_targets = []
#     flat_preds =[]
#     for i in range(targets.shape[0]):
#         target_adj_m = one_hots(targets[i][:lengths[i]])
#         pred_adj_m = one_hots(preds[i][:lengths[i]])

#         flat_targets.extend(target_adj_m.flatten().tolist())
#         flat_preds.extend(pred_adj_m.flatten().tolist())

#     No_Link_p, Link_p = precision_score(flat_targets, flat_preds, labels=[0,1], average=None)
#     No_Link_r, Link_r = recall_score(flat_targets, flat_preds, labels=[0,1], average=None)

#     Link_f1 = 2 * ((Link_p * Link_r) / (Link_p + Link_r))
#     No_Link_fi = 2 * ((No_Link_p * No_Link_r) / (No_Link_p + No_Link_r))

#     link_f1 = (Link_f1 + No_Link_fi) / 2

#     return {
#             "link-f1":link_f1,
#             "LINK-precision":Link_p,
#             "LINK-recall":Link_r,
#             "LINK-f1":Link_f1,
#             "NO-LINK-precision":No_Link_p,
#             "NO-LINK-recall":No_Link_r,
#             "NO-LINK-f1":No_Link_fi,
#             }


