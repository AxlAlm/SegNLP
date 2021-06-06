


# # {
# #     "name": "lark",
# #     "host": "lark.clasp.gu.se",
# #     "protocol": "sftp",
# #     "username": "axlalm",
# #     "remotePath": "/home/axlalm/SegNLP",
# #     "uploadOnSave": true,
# #     "privateKeyPath": "/Users/xalmax/.ssh/id_rsa"
# #     }

# # {
# #     "python.pythonPath": "/Users/xalmax/opt/anaconda3/envs/segnlp/bin/python"
# # }

# from segnlp import Pipeline
# from segnlp.datasets.am import PE
# from segnlp.models import JointPN
# from segnlp.features import GloveEmbeddings
# from segnlp.features import SegPos

# from segnlp.features import BOW

# exp = Pipeline(
#                 id="jp_pe",
#                 dataset=PE( 
#                             tasks=["label", "link"],
#                             prediction_level="seg",
#                             sample_level="paragraph",
#                             ),
#                 model = JointPN,
#                 metric = "default_segment_metric",
#                 features =[
#                             GloveEmbeddings(),
#                             SegPos(),
#                             BOW()
#                             ],
#             )

# hps = {
#         "general":{
#                 "optimizer": "Adam",
#                 "lr": 0.001,
#                 "batch_size": 16,
#                 "max_epochs":4000,
#                 "patience": 15,
#                 "task_weight": 0.5,
#                 },
#         "Agg":{
#                 "mode":"mix",
#                 },
#         "LLSTM": {  
#                     "dropout":0.9,
#                     "hidden_size": 256,
#                     "num_layers":1,
#                     "bidir":True,
#                     },
#         "Pointer": {
#                     #"dropout":0.0,
#                     "hidden_size":512,
#                     }
#         }

# best_hp = exp.train(
#                         hyperparamaters = hps,
#                         n_random_seeds=1,
#                         ptl_trn_args=dict(
#                                             gpus=[1]
#                                         ),
#                         monitor_metric="val_LINK-f1"
#                         )

# exp1_scores, exp1_outputs = exp.test()
# # exp2_scores, exp2_outputs = exp.test(seg_preds="/tmp/seg_preds.csv")


from segnlp.metrics import overlap_metric
from segnlp.metrics import default_segment_metric
import pandas as pd
import numpy as np

task_labels = {
                "label":["MajorClaim",  "Claim", "Premise",],
                "link_label": ["None", "attack", "support"]
                } 

df2 = pd.DataFrame({
    "T-seg_id": [-1,-1,1,1,1,1,1,1,-1,-1,-1,2,2,2,2,2,-1,-1,-1,-1,-1,3,3,3],
    "seg_id":   [-1,-1,-1,-1,1,1,1,1,-1,-1,-1,2,2,2,2,2,0,3,3,3,-1,-1,-1,-1],
    "link":     [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0],
    "T-link":   [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0],
    "T-label":    [None,None,"Premise","Premise","Premise","Premise","Premise","Premise",None,None,None,"Premise","Premise","Premise","Premise","Premise",None,None,None,None,None,"MajorClaim","MajorClaim","MajorClaim"],
    "label":  [None,None,None,None,"Claim","Claim","Claim","Claim",None,None,None,"Premise","Premise","Premise","Premise","Premise",None,"MajorClaim","MajorClaim","MajorClaim",None,None,None,None],
    "T-link_label": [None,None,"attack","attack","attack","attack","attack","attack",None,None,None,"attack","attack","attack","attack","attack",None,"None","None","None",None,None,None,None],
    "link_label": [None,None,None,None,"support","support","support","support",None,None,None,"attack","attack","attack","attack","attack",None,"None","None","None",None,None,None,None],
})
df2  = df2.replace(-1,np.nan)


print(overlap_metric(df2, task_labels))