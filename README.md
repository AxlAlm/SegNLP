


# SegNLP

A NLP framework for Neural Networks for segmentation and linking tasks such as Argument Mining and Named Entity Recognition and Linking.

The goal for the network is to:

1) Collect layers and techniques in the form of torch.nn.Modules which anyone can import and use.

2) Provide a easy and quick way to build models for Segmentation and Linking tasks.

3) Provide a unified framework for preprocessing, training, metrics and evaluation which used the currently recommended techniques in research and tech.


Long term goals is to:

1) support more datasets and an easy way to add costum datasets

2) support automated model building by hypertuning the selection and combination of layers

3) add more layers


## Setup and Installation

- clone repo
- install the packages using environment.yml or requirements.txt.

NOTE! only tested with python==3.7.4 

## What is SegNLP

SegNLP is a higher order API providing everything you need to build, train and evaluate models solving 1 or more of the following tasks:

1) segmentation (seg): Identifying segments in text
2) labeling (label): Labeling/classifying segments into a type
3) linking (link): linking segments to eachother
4) link labeling (link_label): labeling the link between two segments

Under this higher order abstraction of NLP tasks falls for example tasks such as Argument Mining, Named Entity Recognition and Linking/relation extraction.

Currently, only Argument Mining is supported and tested.

SegNLP will provide preprocessing, data handling, logging, training, evaluation, metric, model building tools as well as modular layers specific to a task or a type of technique for encoding, embedding, aggregation, attention, dropout etc..



## Overview

The diagram below gives a very simplified overview over the APIs and core Classes.

![](/docs/api_overview.png)

The "only" things you have to do to use SegNLP is to:

1) select a dataset and define the configuration of it

2) create a SegModel

3) set up a pipeline

4) run Pipeline.train() to tune hyperparamaters and rank them

5) run Pipeline.test() to test the best hyperparamaters on the test set

6) TBA compare different models with eachother 



## how does SegNLP.Pipeline work?

TBA


## how does SegNLP.SegModel work?

TBA


## what layers are there?

TBA


## How is Evaluation done?

TBA


## Replication study in Argument Mining


## An example


##### Pick a dataset

```python
from segnlp.datasets.am import PE

# setting up the dataset
# some dataset such as PE allows for different configurations. E.g. which tasks to perform or which level a sample is on
pe = PE(
    tasks=["seg+label", "link", "link_label"],
    prediction_level="token",
    sample_level="document",
    )
```

##### Setup a pipeline/experiment

```python
from segnlp import Pipeline

# setting up the pipeline  
# the pipeline will create folder, preprocess the whole dataset and do all preparation needed to start training.
exp = Pipeline(
                id = "lstm_er_pe", # will create folder at with id as name at ~/.segnlp/<id>/ 
                dataset = pe,
                metric = "overlap_metric", # settting metric, can be found in segnlp.metrics
                model = "LSTM_ER",
                #overwrite = True, # will remove folder at ~/.segnlp/<id> and create new one
                )


# Another example!
#NOTE! for some experiments need pretrained features. These features often take a long time to extract which means
# we can save a lot of time during training if we preprocess them and save them to disk. The Pipeline takes care of that!
exp = Pipeline(
                id="pe_lstm_crf_seg+label",
                dataset=PE( 
                            tasks=["seg+label"],
                            prediction_level="token",
                            sample_level="sentence",
                            ),
                pretrained_features =[
                            GloveEmbeddings(),
                            FlairEmbeddings(),
                            BertEmbeddings(),
                            ],
                model = "LSTM_CRF",
                metric = "default_token_metric"
            )
```



```python
# setup some hyperparamaters
# hyperparamaters are organized in layers while the more generic/general hyperparamaters are under "general"
# some hyperparamaters such as "optimizer" and "lr_scheduler", or "activation" will jack right into pytorch, i.e.
#  what ever passed to "optimizer" will be looked for in "torch.optim".

from segnlp.resources.vocab import bnc_vocab
import gensim.downloader as api

hps = {
        "general":{
                "optimizer": "Adam",
                "optimizer_kwargs": {
                                    "lr": 0.001,
                                    "weight_decay": 1e-5, #for L2 reg
                                    "eps": 1e-6, 
                                    },
                "batch_size": 1,
                "max_epochs":100,
                "patience": 10,
                "task_weight": 0.5,
                "use_target_segs_k": 10, # sampling
                "freeze_segment_module_k": 25,
                },
        "WordEmb":{
                    "vocab": bnc_vocab(size = 10000),
                    "path_to_pretrained": api.load("glove-wiki-gigaword-50", return_path=True)        
                    },

       "LSTM": {   
                    "input_dropout": 0.5,
                    "dropout":0.5,
                    "hidden_size": 100,
                    "num_layers":1,
                    "bidir":True,
                    },
        "BigramSeg": {
                        "hidden_size": 100,
                        "dropout": 0.5
                },
        "Agg":{
                "mode":"mean",
                },
        "DepTreeLSTM": {
                        "dropout":0.5,
                        "hidden_size":100,
                        "bidir":True,
                        "mode": "shortest_path",
                        },
        "LinearPairEnc": {
                        "hidden_size":100,
                        "activation": "Tanh",
                        },
        "DirLinkLabeler": {
                            "dropout": 0.3,
                            "match_threshold": 0.5,
                        }
        }
```



##### Train some models

```python

# Will train models for each random seed and each set of hyperparamaters given. We train over n random seeds so 
# we can make sure comparison of hyperparamaters can be statistically significance. We follow the method outlined here [REF]
# All outputs are also logged to ~/.segnlp/<id>/logs/<n>.log
exp.train(
        hyperparamaters = hps,
        n_random_seeds=6,
        monitor_metric="f1-0.5",
        #overfit_n_batches = 1
        )
```

##### Visualizing the trained models


##### Testing some models



### Mentions / References

This framework is built upon the following python libs and segnlp would not be what it is without these:

- Pytroch
- Pandas
- Numpy
- FlairNLP
- Gensim


The same goes for all the reserch in Argument Mining:

TBA

### TODO

-  make pip package
