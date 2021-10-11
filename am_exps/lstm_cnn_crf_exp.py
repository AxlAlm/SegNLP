
# genism
import gensim.downloader as api

# segnlp
from segnlp import Pipeline
from segnlp.datasets import PE

# models
from am_exps.models.lstm_cnn_crf import LSTM_CNN_CRF


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


def lstm_cnn_crf(sample_level:str, mode:str, gpu = None):
                
    dataset = PE( 
                    tasks=["seg+label+link+link_label"],
                    prediction_level="token",
                    sample_level=sample_level,
                    )


    pipe = Pipeline(
                id=f"lstm_cnn_crf_{sample_level}",
                dataset = dataset,
                model = LSTM_CNN_CRF,
                metric = "overlap_metric",
            )


    if mode == "train" or mode == "both":
        pipe.train(
                    hyperparamaters = hps,
                    monitor_metric="0.5-f1-micro",
                    gpus = gpu
                    )


    if mode == "test" or mode == "both":
        pipe.test(
                    monitor_metric = "0.5-f1-micro",
                    batch_size = 10,
                    gpus = gpu
                    )