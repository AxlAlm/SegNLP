
import sys
sys.path.insert(1, '../../')

from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.pretrained_features import GloveEmbeddings

import flair, torch
flair.device = torch.device('cpu') 


            
exp = Pipeline(
                id="lstm_er_pe",
                dataset=PE(
                    tasks=["seg+label", "link", "link_label"],
                    prediction_level="token",
                    sample_level="document",
                ),
                metric = "overlap_metric",
                model = "LSTM_ER",
                pretrained_features = [
                                GloveEmbeddings(),
                            ],
                overwrite = True,
                )


hps = {
        "general":{
                "optimizer": "Adam",
                "lr": 0.001,
                "batch_size": 32,
                "max_epochs":100,
                "patience": 10,
                "task_weight": 0.5,
                "seg_gts_k": 10,
                },

       "LSTM": {   
                    "input_dropout": 0.5,
                    "dropout":0.5,
                    "hidden_size": 100,
                    "num_layers":1,
                    "bidir":True,
                    },
        "BigramSeg": {
                        "hidden_size": 512,
                        "activation": "Sigmoid"
                },
        "Agg":{
                "mode":"mean",
                },
        "DepTreeLSTM": {
                        "dropout":0.5,
                        "hidden_size":100,
                        "bidir":True,
                        "graph_buid_type": 0,
                        "sub_graph_type": 0,
                        },

        "LinearPair": {
                        "hidden_size":100,
                        "activation": "Tanh",
                        },
        "DirLinkLabeler": {
                            "dropout": 0.5,
                            "match_threshold": 0.5,
                        }
        }



best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        monitor_metric="val_f1-50%"
                        )



