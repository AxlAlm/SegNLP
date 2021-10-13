

# segnlp
from segnlp import Pipeline
from segnlp.datasets import PE
from segnlp.pretrained_features import GloveEmbeddings
from segnlp.pretrained_features import FlairEmbeddings
from segnlp.pretrained_features import BertEmbeddings

# from models
from am_exps.models.lstm_crf import LSTM_CRF


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
                "gradient_clip_val": 5.0
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
                }
    }



def lstm_crf(sample_level:str, mode:str, gpu = None):

    dataset  = PE( 
                tasks=["seg"],
                prediction_level="token",
                sample_level=sample_level,
                )

    pipe = Pipeline(
                    id=f"lstm_crf_{sample_level}",
                    dataset= dataset,
                    pretrained_features =[
                                GloveEmbeddings(),
                                FlairEmbeddings(),
                                BertEmbeddings(),
                                ],
                    model = LSTM_CRF,
                    metric = "default_token_metric"
                )


    if mode == "train" or mode == "both":
        pipe.train(
                hyperparamaters = hps,
                monitor_metric = "f1",
                gpus = gpu
                )


    if mode == "test" or mode == "both":
        pipe.test(
                    monitor_metric = "f1",
                    batch_size = 32,
                    gpus = gpu
                    )
