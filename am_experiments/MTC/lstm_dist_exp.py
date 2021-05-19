

import sys
sys.path.insert(1, '../../')
import pandas as pd

pd.set_option('display.max_rows', 100)

from segnlp import Pipeline
from segnlp.datasets.am import MTC
from segnlp.nn.models.am import LSTM_DIST
from segnlp.features import ELMoEmbeddings
from segnlp.features import UnitPos, BOW
from segnlp.nn.default_hyperparamaters import get_default_hps

import flair, torch
flair.device = torch.device('cpu') 

exp = Pipeline(
                project="LSTM_DIST",
                dataset= MTC( 
                            tasks=["label", "link", "link_label"],
                            prediction_level="unit",
                            sample_level="document",
                            ),
                features =[
                            ELMoEmbeddings(),
                            UnitPos(),
                            BOW()
                            ],
                model = LSTM_DIST,
                evaluation_method="cross_validation",
                other_levels = ["am"]

            )

print(exp.id)

hps = {
    "optimizer": "adam",
    "lr": 0.001,
    "hidden_dim": 256,
    "num_layers": 1,
    "bidir": True,
    "batch_size": 16,
    "max_epochs": 500,
    "loss_weight": 0.25,
    "input_dropout": 0.1,
    "lstm_dropout" : 0.9,
    "output_dropout": 0.9,
    "patience": 10
    }

best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                           gpus=[1]
                                        ),
                        )

# exp1_scores, exp1_outputs = exp.test()
#seg_pred = pd.read_csv("/tmp/pred_segs.csv")
# print(exp1_scores)
#exp2_scores, exp2_outputs = exp.test(seg_preds=seg_pred)
