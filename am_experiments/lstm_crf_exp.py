
import segnlp
from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.nn.models.general import LSTM_CRF
from segnlp.features import GloveEmbeddings, FlairEmbeddings, BertEmbeddings
from segnlp.nn.default_hyperparamaters import get_default_hps

from segnlp.utils import random_ints, set_random_seed
import numpy as np

from pprint import pprint


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
best_hp = exp.fit(
                        hyperparamaters = hps,
                        random_seed=2019,
                        ptl_trn_args=dict(
                                            gpus=[0]
                                        )
                        )

scores, output_df = exp.test()
output_df.to_csv("/tmp/pred_segs.csv")

