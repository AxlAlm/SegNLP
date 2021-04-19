from segnlp.datasets import PE
from segnlp import Pipeline
from segnlp.features import DummyFeature, OneHots
from segnlp.nn.models import LSTM_CRF
from segnlp.nn.default_hyperparamaters import get_default_hps

from segnlp.utils import list_experiments, exp_summery
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# list_experiments()
# exp_summery("65173953")

# import multiprocessing

# print("CPU COUNT", multiprocessing.cpu_count())

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
        
hps = get_default_hps(LSTM_CRF.name())
hps["max_epochs"] = 1
hps["lr"] = [0.001, 0.005] 

exp.fit(
        hyperparamaters=hps,
        ptl_trn_args=dict(
                            overfit_batches=0.1
                            )
        )

exp.test()


