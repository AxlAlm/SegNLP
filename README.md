


## Setup and Installation

- clone repo
- install the packages in evironment.yml

## Get Started

##### Pick a dataset

```python
from hotam.datasets import PE

pe = PE()
```


##### Prepare dataset for experiment

```python
from hotam import Pipeline
from hotam.features import GloveEmbeddings, BOW

exp = Pipeline(
                project="debugging",
                tasks=["label", "link","link_label"],
                dataset=PE(),
                prediction_level="unit",
                sample_level="paragraph",
                input_level="document", # same as dataset level
                features = [
                            GloveEmbeddings(),
                            BOW(),
                            ],
                argumentative_markers=True
            )
```

##### pick a model

```python
from hotam.nn.models import LSTM_DIST
```

##### Run an experiment

```python
exp1.fit(
        model=LSTM_DIST,
        hyperparamaters = get_default_hps(LSTM_DIST.name()),
        exp_logger=exp_logger,
        gpus=[1],
        )
```


### Information

- [Datasets](https://github.com/AxlAlm/HotAM/blob/main/docs/datasets.md)
- [Features](https://github.com/AxlAlm/HotAM/blob/main/docs/features.md)
- [Models](https://github.com/AxlAlm/HotAM/blob/main/docs/models.md)
- [Training with ExperimentManager]()
- [Logging]()
- [Database]()
- [Dashboard]()


### Mentions / References

This framework is built upon the following python libs and HotAM would not be what it is without these:

- Pytroch
- Pytroch Lightning
- FlairNLP


The same goes for all the reserch in Argument Mining:

TBA

### TODO

- installing by Pip