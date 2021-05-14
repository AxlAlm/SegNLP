

import sys
sys.path.insert(1, '../')
import pandas as pd
from pprint  import pprint

pd.set_option('display.max_rows', 100)


from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.nn.models.am import LSTM_DIST
from segnlp.features import ELMoEmbeddings
from segnlp.features import UnitPos, BOW
from segnlp.nn.default_hyperparamaters import get_default_hps

import flair, torch
flair.device = torch.device('cpu') 

import numpy as np
exp = Pipeline(
                project="LSTM_DIST",
                dataset=PE( 
                            tasks=["label", "link", "link_label"],
                            prediction_level="unit",
                            sample_level="document",
                            ),
                features =[
                            #ELMoEmbeddings(),
                            #UnitPos(),
                            #BOW()
                            ],
                model = LSTM_DIST,
                other_levels = ["am"]
            )

hps = get_default_hps(LSTM_DIST.name())
best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                           gpus=[1]
                                        ),
                        #monitor_metric="val-f1"
                        )

exp1_scores, exp1_outputs = exp.test()

#seg_pred = pd.read_csv("/tmp/pred_segs.csv")

# print(exp1_scores)
#exp2_scores, exp2_outputs = exp.test(seg_preds=seg_pred)
