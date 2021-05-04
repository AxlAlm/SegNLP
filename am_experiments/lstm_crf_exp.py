
import sys
sys.path.insert(1, '../')

from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.nn.models.general import LSTM_CRF
from segnlp.features import GloveEmbeddings, FlairEmbeddings, BertEmbeddings
from segnlp.nn.default_hyperparamaters import get_default_hps

from segnlp.utils import random_ints, set_random_seed
import numpy as np

from pprint import pprint

import flair, torch
flair.device = torch.device('cpu') 


exp = Pipeline(
                project="LSTM_CRF",
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
                model = LSTM_CRF
            )

hps = get_default_hps(LSTM_CRF.name())
hps["max_epochs"] = 2
best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=1,
                        ptl_trn_args=dict(
                                            gpus=[3],
                                            gradient_clip_val=5.0
                                        ),
                        )

score, output = exp.test()
output.to_csv("/tmp/pred_segs.csv")

