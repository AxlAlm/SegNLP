
import segnlp
from segnlp import Pipeline
from segnlp.datasets.am import PE

from segnlp.nn.models.general import LSTM_CNN_CRF
from segnlp.features import GloveEmbeddings
from segnlp.nn.default_hyperparamaters import get_default_hps

exp = Pipeline(
                project="lstm_cnn_crf",
                dataset=PE( 
                            tasks=["seg+label+link+link_label"],
                            prediction_level="token",
                            sample_level="document",
                            ),
                features =[
                            GloveEmbeddings(),
                            ],
                encodings=["chars"],
                model = LSTM_CNN_CRF
            )

hps = get_default_hps(LSTM_CNN_CRF.name())
best_hp = exp.hp_tune(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            gpus=[1]
                                        )
                        )

