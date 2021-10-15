

# segnlp
from segnlp import Pipeline
from segnlp.datasets import PE
from segnlp.pretrained_features import GloveEmbeddings

# models
from am_exps.models.lstm_crf_jpnn import LSTM_CRF_JPNN


hps = {
        "general":{
                "optimizer": {
                                "name": "SGD",
                                "lr": 0.1,
                                },
                "lr_scheduler":{ 
                                "name": "ReduceLROnPlateau",
                                "mode": "max",
                                "factor": 0.5,
                                "patience": 3,
                                "min_lr": 0.0001
                                },
                "batch_size": 32,
                "max_epochs":150,
                "patience": 5,
                "gradient_clip_val": 5.0,
                "task_weight": 0.5,
                    },
        "linear_finetuner":{},
        "token_dropout": {
                        "p":0.05
                        },
        "paramater_dropout": {
                        "p":0.5
                        },
        "LSTM": {  
                    "dropout":0.5,
                    "hidden_size": 256,
                    "num_layers":2,
                    "bidir":True,
                    },
        "CRF": {
                },
        "agg":{
                "mode":"mix",
                },
        "seg_bow":{
                "vocab": "BNC_10k",
                "mode": "counts"
                },
        "linear_fc": {
                "hidden_size": 512,
                "activation": "Sigmoid"
                },
        "lstm_encoder": {  
                    "dropout":0.9,
                    "hidden_size": 256,
                    "num_layers":1,
                    "bidir":True,
                    },
        "lstm_decoder": {  
                    "dropout":0.9,
                    "hidden_size": 512,
                    "num_layers":1,
                    "bidir":False,

                    },
        "Pointer": {}
        }



def lstm_crf_jpnn(sample_level:str, mode:str, gpu = None):

    dataset = PE( 
            tasks=["seg", "label", "link"],
            prediction_level="token",
            sample_level=sample_level,
            )

        
    pipe = Pipeline(
                    id=f"lstm_crf_jpnn_{sample_level}",
                    dataset=dataset,
                    model = LSTM_CRF_JPNN,
                    metric = "overlap_metric",
                    pretrained_features =[
                                            GloveEmbeddings(),
                                            ],
                )
    

    if mode == "train" or mode == "both":
        pipe.train(
                hyperparamaters = hps,
                monitor_metric="0.5-f1",
                gpus = gpu
                )


    if mode == "test" or mode == "both":
        pipe.test(
                    monitor_metric="0.5-f1",
                    batch_size=16,
                    gpus = gpu
                    )