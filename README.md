


## Setup and Installation

- clone repo
- install the packages in evironment.yml

## Get Started

##### Pick a dataset

```python
from segnlp.datasets import PE
pe = PE(
        tasks=["label", "link","link_label"],
        prediction_level="unit",
        sample_level="paragraph",
        )
```

##### Prepare dataset for experiment

```python
from segnlp import Pipeline
from segnlp.features import GloveEmbeddings, BOW

exp = Pipeline(
                project="my_project",
                dataset=pe,
                features = [
                            GloveEmbeddings(),
                            BOW(),
                            ],
            )
```

##### pick a model

```python
from segnlp.nn.models import LSTM_DIST
from segnlp.nn.default_hyperparamaters import get_default_hps
hps = get_default_hps(LSTM_DIST.name())
```

##### Run an experiment

```python
from segnlp.nn.default_hyperparamaters import get_default_hps

exp1.fit(
        model=LSTM_DIST,
        hyperparamaters = hps,
        )
```


### Information

- [Datasets](https://github.com/AxlAlm/segnlp/blob/main/docs/datasets.md)
- [Features](https://github.com/AxlAlm/segnlp/blob/main/docs/features.md)
- [Models](https://github.com/AxlAlm/segnlp/blob/main/docs/models.md)
- [Training with ExperimentManager]()
- [Logging]()
- [Database]()
- [Dashboard]()


### Mentions / References

This framework is built upon the following python libs and segnlp would not be what it is without these:

- Pytroch
- Pytroch Lightning
- FlairNLP


The same goes for all the reserch in Argument Mining:

TBA

### TODO

- installing by Pip