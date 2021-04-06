from hotam.datasets import PE
from hotam import Pipeline
from hotam.features import DummyFeature, OneHots
from hotam.nn.models import LSTM_CRF
from hotam.nn.default_hyperparamaters import get_default_hps


exp = Pipeline(
                project="debugging",
                dataset=PE(
                            tasks=["seg+label"],
                            prediction_level="token",
                            sample_level="sentence",
                            ),
                #encodings=["pos", "deprel", "dephead"],
                features=[
                            DummyFeature(),
                                ]
            )

exp.fit(
        model=LSTM_CRF,
        hyperparamaters=get_default_hps(LSTM_CRF.name()),
        # exp_logger=exp_logger,
        # ptl_trn_args=dict(
        #                     gpus=[2]
        #                     )
        )
