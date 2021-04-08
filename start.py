from hotam.datasets import PE
from hotam import Pipeline
from hotam.features import DummyFeature, OneHots
from hotam.nn.models import LSTM_CRF
from hotam.nn.default_hyperparamaters import get_default_hps

from hotam.utils import list_experiments, exp_summery
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

list_experiments()
exp_summery("65173953")

import multiprocessing

print("CPU COUNT", multiprocessing.cpu_count())

exp = Pipeline(
                project="debugging",
                model=LSTM_CRF,
                dataset=PE(
                            tasks=["seg+label"],
                            prediction_level="token",
                            sample_level="sentence",
                            ),
                features=[
                            DummyFeature(),
                                ]
            )
        
# hps = get_default_hps(LSTM_CRF.name())
# hps["max_epochs"] = 1
# hps["lr"] = [0.001, 0.005] 

# exp.fit(
#         hyperparamaters=hps,
#         ptl_trn_args=dict(
#                             overfit_batches=0.1
#                             )
#         )

exp.test()


