# HotAM

<p align="center">
  <img src="https://github.com/AxlAlm/HotAM/blob/main/HOTAM_LOGO.png?raw=true" alt="hot jam dam"/>
</p>

## About

HotAM is a Python Framework for Argument Mining.

## Setup and Installation

- clone repo
- install the packages in evironment.yml
- install MongoDB (if you want to log to mongodb and used dashboard)

## Get Started

##### Pick a dataset

```python
from hotam.datasets import PE

pe = PE()
```

##### Explore a bit

```python

pe.example()
pe.stats()
```

##### Prepare dataset for experiment

```python
from hotam.features import Embeddings

pe.setup(
    tasks=["seg"],
    sample_level="document",
    prediction_level="token",	
    encodings=["pos"],
    features=[
    		Embeddings("glove")
    		],
	)
```

##### pick a model

```python
from hotam.nn.models import LSTM_CRF
```

##### setup logging 

```python
from hotam.database import MongoDB
from hotam.loggers import MongoLogger

db = MongoDB()
exp_logger = MongoLogger(db=db)
```

##### Run an experiment

```python
from hotam import ExperimentManager

M = ExperimentManager()
M.run( 
    project="my_project",
    dataset=pe,
    model=LSTM_CRF,
    monitor_metric="val-seg-f1",
    progress_bar_metrics=["val-seg-f1"],
    exp_logger=exp_logger
    )
```

##### Start Dashboard to view experiment live and view past experiments
this can be done any time as it will be running in sync with ExperimentManager.run()

```python
from hotam.dashboard import Dashboard

Dashboard(db=db).run_server(
			    port=8050,
			    )
```

![](https://github.com/AxlAlm/HotAM/blob/main/hotam-modules.png)


### Information

- [Datasets](https://github.com/AxlAlm/HotAM/blob/main/docs/datasets.md)
- [Features](https://github.com/AxlAlm/HotAM/blob/main/docs/features.md)
- [Models](https://github.com/AxlAlm/HotAM/blob/main/docs/models.md)
- [Training with ExperimentManager]()
- [Logging]()
- [Database]()
- [Dashboard]()


### Mentions

This framework is built upon the following python libs and HotAM would not be what it is without these

- Pytroch
- Pytroch Lightning
- FlairNLP
- Dash

### References

TBA


### TODO

- installing by Pip

