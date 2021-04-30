
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


#exp.dataset[np.random.randint(6000, size=(32,))]
#exp.dataset[np.arange(500,532,1)]
#print(exp.dataset.splits)

segnlp.settings["dl_n_workers"] = 0
hps = get_default_hps(LSTM_CRF.name())
best_hp = exp.hp_tune(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            gpus=[0]
                                        )
                        )

# exp.test()

