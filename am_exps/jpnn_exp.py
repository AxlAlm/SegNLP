

# segnlp
from segnlp import Pipeline
from segnlp.datasets import PE
from segnlp.pretrained_features import GloveEmbeddings

# models
from am_exps.models.jpnn import JointPointerNN


hps = {
        "general":{
                "optimizer": "Adam",
                "lr": 0.001,
                "batch_size": 16,
                "max_epochs":4000,
                "patience": 10,
                "task_weight": 0.5,
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



def jpnn(sample_level:str, mode:str, gpu = None):

    dataset = PE( 
            tasks=["label", "link"],
            prediction_level="seg",
            sample_level=sample_level,
            )

        
    pipe = Pipeline(
                    id=f"jpnn_{sample_level}",
                    dataset=dataset,
                    model = JointPointerNN,
                    metric = "default_segment_metric",
                    pretrained_features =[
                                            GloveEmbeddings(),
                                            ],
                )
    

    if mode == "train" or mode == "both":
        pipe.train(
                hyperparamaters = hps,
                monitor_metric="link-f1",
                gpus = gpu
                )


    if mode == "test" or mode == "both":
        pipe.test(
                    monitor_metric="link-f1",
                    batch_size=16,
                    gpus = gpu
                    )