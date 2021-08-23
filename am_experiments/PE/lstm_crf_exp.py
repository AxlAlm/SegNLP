
import sys
sys.path.insert(1, '../../')

from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.features import GloveEmbeddings, FlairEmbeddings, BertEmbeddings

# import flair, torch
# flair.device = torch.device('cuda:0') 

exp = Pipeline(
                id="lstm_crf_pe_seg",
                dataset=PE( 
                            tasks=["seg"],
                            prediction_level="token",
                            sample_level="sentence",
                            ),
                features =[
                            GloveEmbeddings(),
                            FlairEmbeddings(),
                            BertEmbeddings(),
                            ],
                model = "LSTM_CRF",
                metric = "default_token_metric"
            )


hps = {
        "general":{
                "optimizer": "SGD",
                "lr": 0.1,
                "batch_size": 10,
                "max_epochs":200,
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
                                            #gpus=[1],
                                            gradient_clip_val=5.0
                                        ),
                        )

# score, output = exp.test()
# output.to_csv("/tmp/pred_segs.csv")

