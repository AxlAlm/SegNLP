

from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.pretrained_features import DummyFeature


exp = Pipeline(
                id="pe_lstm_crf_seg+label",
                dataset=PE( 
                            tasks=["seg+label"],
                            prediction_level="token",
                            sample_level="sentence",
                            ),
                pretrained_features =[
                            DummyFeature()
                            ],
                model = "LSTM_CRF",
                metric = "default_token_metric"
            )


hps = {
        "general":{
                "optimizer": "SGD",
                "lr": 0.1,
                "batch_size": 10,
                "max_epochs":2,
                "patience": 5,
                },
        "CharEmb": {  
                    "emb_size": 30,
                    "n_filters": 20,
                    "kernel_size": 3,
                    },
        "LSTM": {  
                    "dropout":0.5,
                    "hidden_size": 150,
                    "num_layers":1,
                    "bidir":True,
                    },
        "CRF": {
                "dropout": 0.5,
                }
        }


best_hp = exp.train(
                    hyperparamaters = hps,
                    n_random_seeds=6,
                    ptl_trn_args=dict(
                                    gpus = [1],
                                    gradient_clip_val = 5.0,
                                    overfit_batches = 0.1
                                    ),
                    )

# score, output = exp.test()
# output.to_csv("/tmp/pred_segs.csv")

