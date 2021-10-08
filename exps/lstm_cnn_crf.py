

import argparse
from segnlp import Pipeline
from segnlp.datasets.am import PE
import gensim.downloader as api


hps = {
        "general":{
                "optimizer": {
                        
                                "name":"Adadelta",
                                "lr": 0.01,
                                "weight_decay": 0.05
                                },
                "batch_size": 10,
                "max_epochs":1000,
                "patience": 5,
                "gradient_clip_val": 5.0
                },
        "dropout": {
                        "p": 0.5
                },
        "word_embs": {
                        "vocab": "BNC_10k",
                        "path_to_pretrained" : api.load("glove-wiki-gigaword-50", return_path=True)
                        },
        "CharEmb": {  
                    "embedding_dim": 30,
                    "n_filters": 50,
                    "kernel_size": 3,
                    "dropout": 0.5
                    },
        "LSTM": {  
                    "hidden_size": 200,
                    "num_layers":1,
                    "bidir":True,
                    "weight_init": "xavier_uniform_" # glorot
                    },
        "CRF": {
                }
        }



def run():
        exp = Pipeline(
                        id="pe_lstm_cnn_crf_doc",
                        dataset=PE( 
                                tasks=["seg+label+link+link_label"],
                                prediction_level="token",
                                sample_level="document",
                                ),
                        model = "LSTM_CNN_CRF",
                        metric = "overlap_metric",
                )



        best_hp = exp.train(
                                hyperparamaters = hps,
                                n_random_seeds=1,
                                monitor_metric="f1-0.5-micro",
                                gpus = [2]
                                )