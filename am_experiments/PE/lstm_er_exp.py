
import sys
sys.path.insert(1, '../')
import pandas as pd
from pprint  import pprint

pd.set_option('display.max_rows', 100)

from segnlp.datasets.am import PE
from segnlp import Pipeline
from segnlp.features import GloveEmbeddings, OneHots
from segnlp.nn.models.general import LSTM_ER
from segnlp.nn.default_hyperparamaters import get_default_hps

import flair, torch
flair.device = torch.device('cpu') 


exp = Pipeline(
                project="debugging",
                dataset=PE(
                    tasks=["seg+label", "link", "link_label"],
                    prediction_level="token",
                    sample_level="document",
                ),
                model=LSTM_ER,
                encodings=["pos", "deprel", "dephead"],
                features=[
                                GloveEmbeddings(),
                                OneHots("pos"),
                                OneHots("deprel")
                            ]
                )
                
hps = get_default_hps(LSTM_ER.name())


with torch.autograd.set_detect_anomaly(True):
    best_hp = exp.train(
                            hyperparamaters = hps,
                            n_random_seeds=6,
                            ptl_trn_args=dict(
                                            gpus=[1]
                                            ),
                            )