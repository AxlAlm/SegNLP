
import sys
sys.path.insert(1, '../../')

from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.features import GloveEmbeddings

import flair, torch
flair.device = torch.device('cpu') 


exp = Pipeline(
                id="pe_lstm_cnn_crf",
                dataset=PE( 
                            tasks=["seg+label+link+link_label"],
                            prediction_level="token",
                            sample_level="document",
                            ),
                features =[
                            GloveEmbeddings(),
                            ],
                encodings=["chars"],
                model = "LSTM_CNN_CRF",
                metric = "overlap_metric",
            )

hps = {
        "general":{
                "optimizer": "SGD",
                "lr": 0.001,
                "batch_size": 10,
                "max_epochs":200,
                "patience": 5,
                "task_weight": 0.5,
                },
        "CharEmb": {  
                    "emb_size": 30,
                    "n_filters": 20,
                    "kernel_size": 3,
                    },
        "LSTM": {  
                    "dropout":0.5,
                    "hidden_size": 150,
                    "num_layers":1,
                    "bidir":True,
                    },
        "CRF": {
                "dropout": 0.5,
                }
        }

best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=1,
                        ptl_trn_args=dict(
                                            gpus=[1],
                                            #gradient_clip_val=5
                                        )
                        )

exp1_scores, exp1_outputs = exp.test()
