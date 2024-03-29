

# gensim
import gensim.downloader as api

# segnlp
from segnlp import Pipeline
from segnlp.datasets import PE

# models
from am_exps.models.lstm_er import LSTM_ER

# Hyperparamaters
# one set of hyperparamaters per layer + general hyperaparamaters
hps = {
        "general":{
                "optimizer": {
                                "name":"Adam",
                                "lr": 0.001,
                                "weight_decay":1e-5,
                                "eps": 1e-6
                                },
                "batch_size": 1,
                "max_epochs":100,
                "patience": 10,
                "use_target_segs_k": 10, 
                "freeze_segment_module_k": 25,
                "gradient_clip_val": 10.0
                },
        "dropout": {
                        "p":0.5
                },
        "output_dropout": {
                        "p": 0.3
                        },
        "word_embs": {
                        "vocab": "BNC_10k",
                        "path_to_pretrained" : api.load("glove-wiki-gigaword-50", return_path=True)
                        },
        "pos_embs":{
                        "vocab": "Pos",
                        "embedding_dim":25,
                        "weight_init": "uniform_" #random from uniform
                },
        "dep_embs":{
                        "vocab": "Pos",
                        "embedding_dim":25,
                        "weight_init": "uniform_" #random from uniform
                },
       "LSTM": {   
                        "hidden_size": 100,
                        "num_layers":1,
                        "bidir":True,
                        "weight_init": "uniform_" #random from uniform
                    },
        "BigramSeg": {
                        "hidden_size": 100,
                        "weight_init": "uniform_" #random from uniform
                },
        "Agg":{
                "mode":"mean",
                },
        "DepTreeLSTM": {
                        "hidden_size":100,
                        "bidir":True,
                        "mode": "shortest_path",
                        "weight_init": "uniform_" #random from uniform
                        },
        "LinearPairEnc": {
                        "hidden_size":100,
                        "activation": "Tanh",
                        "weight_init": "uniform_" #random from uniform
                        },
        "DirLinkLabeler": {
                            "match_threshold": 0.3,
                        }
        }


def lstm_er(sample_level:str, mode:str, gpu = None):

    dataset = PE(
                    tasks=["seg+label", "link", "link_label"],
                    prediction_level="token",
                    sample_level=sample_level,
                )


    pipe = Pipeline(
                    id = f"lstm_er_{sample_level}",
                    dataset = dataset,
                    metric = "overlap_metric",
                    model = LSTM_ER,
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
                    batch_size = 1,
                    gpus = gpu
                    )
