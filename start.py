from hotam.datasets import PE
from hotam import Pipeline
from hotam.features import DummyFeature, OneHots
from hotam.nn.models import LSTM_ER
from hotam.nn.default_hyperparamaters import get_default_hps

pe = PE()

exp = Pipeline(
    project="debugging",
    tasks=["seg+label", "link", "link_label"],
    dataset=PE(),
    prediction_level="token",
    sample_level="document",
    input_level="document",  # same as dataset level
    encodings=["pos", "deprel", "dephead"],
    features=[
        DummyFeature(),
        OneHots("pos"),
        OneHots("deprel")
    ])

exp.fit(
    model=LSTM_ER,
    hyperparamaters=get_default_hps(LSTM_ER.name()),
    # exp_logger=exp_logger,
    gpus=[3],
)
