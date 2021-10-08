

import gensim.downloader as api
from segnlp import Pipeline
from segnlp.datasets.am import PE



# Hyperparamaters
# one set of hyperparamaters per layer + general hyperaparamaters

hps = {
        "general":{
                "optimizer": {
                                "name":"Adam",
                                "lr": 0.001,
                                "weight_decay":1e-5
                                },
                "batch_size": 1,
                "max_epochs":100,
                "patience": 10,
                "task_weight": 0.5,
                "use_target_segs_k": 10, # sampling
                "freeze_segment_module_k": 25, # Havent tested this fully yet. (Next on the list)
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
                        "hidden_size": 512,
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



def run(self, dataset:str, mode:str):


    # setting up the pipeline      
    pipe = Pipeline(
                    id="lstm_er_pe_doc", # will create folder at with id as name at ~/.segnlp/<id>/ 
                    dataset=PE(
                        tasks=["seg+label", "link", "link_label"],
                        prediction_level="token",
                        sample_level="document",
                    ),
                    metric = "overlap_metric", # settting metric, can be found in segnlp.metrics
                    model = "LSTM_ER",
                    #overwrite = True, # will remove folder at ~/.segnlp/<id> and create new one
                    )


        if mode == "train" or  mode == "both":
            # Will train models for each random seed and each set of hyperparamaters given. All models are saved
            # All outputs are also logged to ~/.segnlp/<id>/logs/<n>.log but havent fully test this either.
            pipe.train(
                                    hyperparamaters = hps,
                                    n_random_seeds=6,
                                    monitor_metric="0.5-f1-micro",
                                    gpus = [1]
                                    )


        if mode == "test" or  mode == "both":
            pipe.test()