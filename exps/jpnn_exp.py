






from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.pretrained_features import GloveEmbeddings


hps = {
        "general":{
                "optimizer": "Adam",
                "lr": 0.001,
                "batch_size": 16,
                "max_epochs":1,
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



def jpnn(dataset:str, mode:str, gpu = None):


    if dataset == "PE":
        dataset = PE( 
                tasks=["label", "link"],
                prediction_level="seg",
                sample_level="paragraph",
                )

        
    pipe = Pipeline(
                    id="jp_nn_pe_para",
                    dataset=dataset,
                    model = "JointPN",
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
                    gpu = gpu
                    )