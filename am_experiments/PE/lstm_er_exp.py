
import sys
sys.path.insert(1, '../../')

from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.models import JointPN
from segnlp.features import GloveEmbeddings
from segnlp.features import SegPos
from segnlp.features import BOW
from segnlp.features import OneHots

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
                encodings = ["pos", "deprel", "dephead"],
                features = [
                                GloveEmbeddings(),
                                OneHots("pos"),
                                OneHots("deprel")
                            ]
                )


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
        "DirLinkLabeler": {
                            "hidden_size":100,
                            "dropout": 0.5
                        }
        }



best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            #gpus=[1]
                                        ),
                        monitor_metric="val_f1"
                        )



