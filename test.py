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
                "max_epochs":150,
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
        "LSTM": {  
                    "dropout":0.5,
                    "hidden_size": 256,
                    "num_layers":2,
                    "bidir":True,
                    },
        "CRF": {
                }
    }



if __name__ == "__main__":

    pe = PE(
            tasks = ["seg"],
            prediction_level = "token",
            sample_level = "paragraph"
    )

    exp = Experiment(
                id = "test",
                dataset = pe,
                metric = "default_token_metric",
            )

    exp.train(
                model = LSTM_CRF,
                hyperparamaters = hps,
                monitor_metric = "f1"
    )