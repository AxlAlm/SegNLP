from segnlp.datasets import PE
from segnlp.experiment import Experiment
from segnlp.models.lstm_crf import LSTM_CRF


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
                "max_epochs": 1,
                "patience": 5,
                "gradient_clip_val": 5.0
                },
        "flair_embeddings": {
                        "embs": "flair+bert+glove"
                    },
        "linear_finetuner":{},
        "token_dropout": {
                        "p":0.05
                        },
        "paramater_dropout": {
                        "p":0.5
                        },
        "lstm": {  
                    "dropout":0.5,
                    "hidden_size": 256,
                    "num_layers":2,
                    "bidir":True,
                    },
        "crf": {
                }
    }



if __name__ == "__main__":

    pe = PE(
            tasks = ["seg"],
            prediction_level = "token",
            sample_level = "sentence"
    )

    exp = Experiment(
                id = "pe_seg_sentence",
                dataset = pe,
                metric = "default_token_metric",
                n_random_seeds = 1
            )

    exp.train(
                model = LSTM_CRF,
                hyperparamaters = hps,
                monitor_metric = "f1"
    )

    exp.test(
                model = LSTM_CRF,
                monitor_metric = "f1"
    )
    