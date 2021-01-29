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

## Overview
![](https://github.com/AxlAlm/HotAM/blob/main/hotam-modules.png)

## Get Started

### Pick a dataset

```python
#Persuasive Essays
from hotam.datasets import PE

pe = PE()
```

### Explore a bit

```python

pe.example()
pe.stats()
```
Note that calling stats() after the step below will change the output.

### Prepare dataset for experiment

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

Lets take a look at the paramaters:

**tasks** decide which task you will test during the experiment. Above we only test "seg" (segmentation of argument components) For complexed task, e.g. combinations of two or more task into one, you add a "_" between. E.g. if we want to combine "seg" and "ac" (argument component classification) we use the task ["seg_ac"]. If we want to do "seg" and "ac" seperately we pass ["seg", "ac"].  Note that the tasks allowed here are decided by the dataset.

**sample_level** decide which level the samples in your experiment will be on. If you set sample level documents, you will be passing documents. If you set sentences you will pass sentence and so on.

**prediction_level** decides which level your model is to be evaluated and expect predictions on. Here we only have two options  "token" and "ac". Passing "token" will make the framework expect predictions for a sample to be on token level and transform samples, encodings and features to match this. For the above configuration our e.g. input embeddings for a sample to the model will be in the following dimension (batch_size, max_nr_tokens). 
If we pass "ac" our framwork will expect us to predict on Argument Components. If we for example change the above configuare from "token" to "ac" our input e.g. embeddings for our model will be in the following shape (batch_size, max_nr_ac_in_doc, max_token_in_acs). Note that some combinations of tasks, sample_level and prediction_level are not allowed as they dont make sense, e.g. task=["seg"] and prediction_level=["ac"] -- segmentation is already given as we are given Argument Components.

**encodings** decides which encodings will be done. If you wish to pass encodings such as characters, words, pos or dephead to your model you decide this here. Encodings are unique IDS.
Note that labels are always encoded

**features** decided the features that will be extracted from the dataset. In above example glove embeddings will be extracted. Multiple features can be passed. All input to features should be a hotam.FeatureModel (read more about features).


### pick a model

HotAm supports a few model. However, you can easliy create your own as model are expected to be a pytorch.nn.Module

```python
from hotam.nn.models import LSTM_CRF
```

### Run an experiment


```python
from hotam import ExperimentManager

M = ExperimentManager()

# then we run an experiment. We chose a project name under which the experiment will be logged
# we chose a model(a torch.nn module). We will use a LSTM_CRF. See hotam/nn/models for more
# we then need so specify which metric we will monitor. Metrics for all tasks etc will be logged but monitor metric
# decides which is the most imporant metric and will come into play if you want for example, early stopping
# progress_bar_metric is only for metric which will be displayed in terminal when running.
M.run( 
    project="test_project",
    dataset=pe,
    model=LSTM_CRF,
    monitor_metric="val-seg_ac-f1",
    progress_bar_metrics=["val-seg_ac-f1"],
    debug_mode=False,
    )
```
**ExperimentManager** takes care of everything we need for training.


## Datasets
Persuasive Essays (LINK)

## Features
TBA

## Models

- LSTM-CRF [paper](https://www.aclweb.org/anthology/W19-4501) [code](https://github.com/AxlAlm/HotAM/blob/main/hotam/nn/models/lstm_crf.py)
- LSTM-CNN-CRF [paper](https://arxiv.org/pdf/1704.06104.pdf) [code](https://github.com/AxlAlm/HotAM/blob/main/hotam/nn/models/lstm_cnn_crf.py)
- LSTM-DIST [paper](https://www.aclweb.org/anthology/P19-1464/) [code](https://github.com/AxlAlm/HotAM/blob/main/hotam/nn/models/lstm_dist.py) 
- JointPointerNN [paper](https://arxiv.org/pdf/1612.08994.pdf) [code](https://github.com/AxlAlm/HotAM/blob/main/hotam/nn/models/joint_pointer_nn.py)

### Results
TBA

## Training
## Logging
## Dashboard
## Reproducing Results


### pre

- install the packages in evironment.yml
- install MongoDB (if you want to log to mongodb and used dashboard)


### how to run

```python
from hotam.datasets import PE
from hotam.features import DummyFeature

# get our data (Persuasive Essays)
pe = PE()

# set up the datset to our exeperiment.
# this means chosing what sample level, what to predict on, which task etc
#
# here we chose our task to be "seg_ac" which means we combine segmention and Argument Component classification to one task
# labels will then be complexed labels of combination of BIO encoding and Argument Types, e.g. B-Premise, B-Claim, O, I-MajorClaim (labels for PE)
# we also set the sample_level to "document" and prediction_level to "token" meaning our samples will be arrays of token per documents and we 
# assume prediction is done on each token.
#
# lastly we add some encodings and features. "pos" encodings are encoded pos-tags and DummyFeature() is defult to randomly generate 100 dim features for tokens
# What ever you put under features will be grouped under their level. e.g. all features where the feature.level == "doc" will be concatenated and then accessed in
# in batches by batch["word_embs"], and if features.level == "doc" under batch["doc_embs"]
pe.setup(
    tasks=["seg_ac"],
    multitasks=[], 
    sample_level="document",
    prediction_level="token",	
    encodings=["pos"],
    features=[
          DummyFeature()
          ],
	)

```

How do we get the data:


```python

# we can get the data for some sample ids by indexing the pe datset
batch = pe[[1,2,3,4,5]]
```


How do we run experiments on our datset?

```python
from hotam import ExperimentManager
from hotam.nn.models import LSTM_CRF

#set up experiment manager, wich takes care of everything we need for training, except the model
M = ExperimentManager()

# then we run an experiment. We chose a project name under which the experiment will be logged
# we chose a model(a torch.nn module). We will use a LSTM_CRF. See hotam/nn/models for more
# we then need so specify which metric we will monitor. Metrics for all tasks etc will be logged but monitor metric
# decides which is the most imporant metric and will come into play if you want for example, early stopping
# progress_bar_metric is only for metric which will be displayed in terminal when running.
M.run( 
    project="test_project",
    dataset=pe,
    model=LSTM_CRF,
    monitor_metric="val-seg_ac-f1",
    progress_bar_metrics=["val-seg_ac-f1"],
    debug_mode=False,
    )


```

if you want to log to mongodb and/or use dashboard
```python

from hotam.database import MongoDB
from hotam.loggers import MongoLogger

db = MongoDB()
exp_logger = MongoLogger(db=db)

M.run( 
    project="test_project",
    dataset=pe,
    model=LSTM_CRF,
    monitor_metric="val-seg_ac-f1",
    progress_bar_metrics=["val-seg_ac-f1"],
    exp_logger=exp_logger
    )
```


#### What is passed to models and what do you need to define?

When running DataSet.setup() you are setting the dimension of the input the model. You do this by adding featuers and setting sample and prediction level.

All feature models have a variable feature_dim which determine the dimension of the feature. This information is collected in the DataSet and passed to all models
as feature2dim. This dict will contain "word_embs" and/or "doc_embs". These two values are the sum of the feature dimension for each feature that is either on Word level or document level, something which is also set in the feature models.

At the forward pass in a model you will be passed a batch. this batch is a dict but a subclass of a dict. This will contain all information you will need, such as:

lengths
word_embs
doc_embs
ids
any encoding (e.g. chars, or pos)



#### Features

All feature models has a extract() function. this allways takes pandas.DataFrame object and return a numpy vector.

For OneHots make sure that you have passed what you want onehot encodings of into the encoding paramater in DataSet.setup(). For example, if we use OneHot("pos") (one hot encodings for POS tags) make sure that the dataset has pos tags encoded (DataSet.setup(encodings=["pos"])).


#### Hyperparamaters and Training configuration

All training is done with Pytorch Lightning and all paramaters to the pytorch lightning Trainer() you can pass to the trainer_args paramater in ExperimentManager.run(). Default trainer_args are located in hotam.manager.plt_trainer_setup.py

Hyperparamaters can also be passed to ExperimentManager.run(). For all model that are inherent in the lib there default hyperparamaters are set in hotam.default_hyperparamaters.



How to run dashboard

TBA

