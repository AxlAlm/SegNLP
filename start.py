
from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.models import JointPN
from segnlp.features import GloveEmbeddings
from segnlp.features import SegPos
from segnlp.features import BOW

exp = Pipeline(
                id="lstm_er_pe",
                dataset=PE( 
                            tasks=["seg+label", "link_label"],
                            prediction_level="seg",
                            sample_level="paragraph",
                            ),
                model = JointPN,
                metric = "default_segment_metric",
                features =[
                            GloveEmbeddings(),
                            SegPos(),
                            #BOW()
                            ],
                encodings = [
                                "words"
                                ]
            )


lstm_er = {
    "seq_lstm_h_size": 100,  # Sequential LSTM hidden size
    "tree_lstm_h_size": 100,  # Tree LSTM hidden size
    "ac_seg_hidden_size": 100,  # Entity recognition layer hidden size
    "re_hidden_size": 100,  # Relation extraction layer hidden size
    "seq_lstm_num_layers": 1,  # Sequential LSTM number of layer
    "lstm_bidirectional": True,  # Sequential LSTM bidirection
    "tree_bidirectional": True,  # Tree LSTM bidirection
    "k": 25,  # hyperparameter for scheduled sampling
    "graph_buid_type": 0,
    "sub_graph_type": 0,
    "dropout": 0.5,
    "optimizer": "adam",
    "lr": 0.001,
    "max_epochs": 300,
    "batch_size": 32,
    "patience":10,
}

hps = {
        "general":{
                "optimizer": "Adam",
                "lr": 0.001,
                "batch_size": 32,
                "max_epochs":300,
                "patience": 10,
                "task_weight": 0.5,
                "k": 25,
                },
        "Agg":{
                "mode":"mean",
                },
        "BigramSeg": {
                        "hidden_size": 512,
                        "activation": "Sigmoid"
                },
        "LSTM": {  
                    "dropout":0.5,
                    "hidden_size": 100,
                    "num_layers":1,
                    "bidir":True,
                    },
        "DepPairingLayer": {
                        "dropout":0.9,
                        "hidden_size":512,
                        "bidir":True
                        }
        }

best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            #gpus=[1]
                                        ),
                        monitor_metric="val_link-f1"
                        )

exp1_scores, exp1_outputs = exp.test()
# exp2_scores, exp2_outputs = exp.test(seg_preds="/tmp/seg_preds.csv")


# from segnlp.metrics import overlap_metric
# from segnlp.metrics import default_segment_metric
# import pandas as pd
# import numpy as np

# task_labels = {
#                 "label":["MajorClaim",  "Claim", "Premise",],
#                 "link_label": ["None", "attack", "support"]
#                 } 

# df2 = pd.DataFrame({
#     "T-seg_id": [-1,-1,1,1,1,1,1,1,-1,-1,-1,2,2,2,2,2,-1,-1,-1,-1,-1,3,3,3],
#     "seg_id":   [-1,-1,-1,-1,1,1,1,1,-1,-1,-1,2,2,2,2,2,0,3,3,3,-1,-1,-1,-1],
#     "link":     [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0],
#     "T-link":   [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0],
#     "T-label":    [None,None,"Premise","Premise","Premise","Premise","Premise","Premise",None,None,None,"Premise","Premise","Premise","Premise","Premise",None,None,None,None,None,"MajorClaim","MajorClaim","MajorClaim"],
#     "label":  [None,None,None,None,"Claim","Claim","Claim","Claim",None,None,None,"Premise","Premise","Premise","Premise","Premise",None,"MajorClaim","MajorClaim","MajorClaim",None,None,None,None],
#     "T-link_label": [None,None,"attack","attack","attack","attack","attack","attack",None,None,None,"attack","attack","attack","attack","attack",None,"None","None","None",None,None,None,None],
#     "link_label": [None,None,None,None,"support","support","support","support",None,None,None,"attack","attack","attack","attack","attack",None,"None","None","None",None,None,None,None],
# })
# df2  = df2.replace(-1,np.nan)


# print(overlap_metric(df2, task_labels))