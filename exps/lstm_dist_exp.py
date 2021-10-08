

# segnlp
from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.pretrained_features import ELMoEmbeddings


hps = {
        "general":{
                "optimizer": {
                                "name": "Adam",
                                "lr": 0.001,
                                },
                "batch_size": 16,
                "max_epochs":500,
                "patience": 10,
                "task_weight": 0.5,
                },
        "embedding_dropout":{
                        "p":0.1,
                        },
        "bow_dropout":{
                        "p":0.5,
                        },
        "output_dropout":{
                        "p":0.5,
                        },                     
        "seg_bow":{
                "vocab": "BNC_10k",
                "mode": "counts",
                },
        "bow_linear":{
                        "out_dim": 512,
                        "weight_init": {
                                        "name":"uniform_",
                                        "a": -0.05, 
                                        "b": 0.05
                                        },
                        "activation" : "sigmoid"
                        },
        "lstms": {   
                "dropout":0.1,
                "hidden_size": 256,
                "num_layers":1,
                "bidir":True,
                "weight_init": "orthogonal_",
                },
        "Pairer": {
                    "mode": ["cat", "multi"],
                    "n_rel_pos": 25,
                    },
        "linear_cf":  {
                        "name":"uniform_",
                        "a": -0.05, 
                        "b": 0.05
                        },
        }


def lstm_dist(dataset:str, mode:str, gpu = None):
                
    if dataset == "PE":
        dataset = PE( 
                    tasks=["label", "link", "link_label"],
                    prediction_level="seg",
                    sample_level="paragraph",
                    )


    pipe = Pipeline(
                    id = "lstm_dist_pe",
                    dataset = dataset,
                    pretrained_features = [
                                    ELMoEmbeddings(),
                                    ],
                    model = "LSTM_DIST",
                    other_levels = ["am"],
                    metric = "default_segment_metric",
                )


    if mode == "train" or mode == "both":
        pipe.train(
                    hyperparamaters = hps,
                    monitor_metric="f1",
                    gpus = gpu
                    )



    if mode == "test" or mode == "both":
        pipe.test(
                    monitor_metric = "f1",
                    batch_size = 16,
                    gpus = gpu
                    )