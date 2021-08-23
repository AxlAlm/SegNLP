

import sys
sys.path.insert(1, '../../')
f


from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.features import ELMoEmbeddings
from segnlp.features import UnitPos, BOW

import flair, torch
flair.device = torch.device('cpu') 

import numpy as np
exp = Pipeline(
                project="LSTM_DIST",
                dataset=PE( 
                            tasks=["label", "link", "link_label"],
                            prediction_level="unit",
                            sample_level="document",
                            ),
                features =[
                            ELMoEmbeddings(),
                            UnitPos(),
                            BOW()
                            ],
                model = "LSTM_DIST",
                other_levels = ["am"]
            )
            

lstm_dist_hps = {
    "optimizer": "adam",
    "lr": 0.001,
    "hidden_dim": 256,
    "num_layers": 1,
    "bidir": True,
    "batch_size": 16,
    "max_epochs": 500,
    "loss_weight": 0.25,
    "input_dropout": 0.1,
    "lstm_dropout" : 0.1,
    "output_dropout": 0.5,
    "patience": 10

}

hps = {
        "general":{
                "optimizer": "Adam",
                "lr": 0.001,
                "batch_size": 16,
                "max_epochs":500,
                "patience": 10,
                "task_weight": 0.5,
                },
       "LSTM": {   
                    "input_dropout": 0.1,
                    "dropout":0.1,
                    "output_dropout": 0.5, 
                    "hidden_size": 256,
                    "num_layers":1,
                    "bidir":True,
                    },
  
        "Pairer": {
                
                    },
        "DirLinkLabeler": {
                            "hidden_size":100,
                            "dropout": 0.5
                        }
        }


best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                           gpus=[1]
                                        ),
                        #monitor_metric="val-f1"
                        )

exp1_scores, exp1_outputs = exp.test()

#seg_pred = pd.read_csv("/tmp/pred_segs.csv")

# print(exp1_scores)
#exp2_scores, exp2_outputs = exp.test(seg_preds=seg_pred)
