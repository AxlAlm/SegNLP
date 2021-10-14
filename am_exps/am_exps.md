
# Argument Mining Experiments

We use SegNLP to replicate 5 SOTA models in Argument Mining.

The tasks we are focusing on are the following four:

1) identifying argument segments (aka "seg")
2) classifying argument segments (aka "label")
3) linking argument segments (aka "link")
3) classifying links between argument segments (aka "link_label")

All models are solving 1 or more of these task.

To the best of our knowledge the only Argument Mining dataset which contain all these four task is Persuasive Essays [REF]

## How to run Experiments

all experiments are run using the following commoand:


python run.py -model <MODEL> -level <LEVEL> -mode <MODE>  -gpu <ID>

selectign a model will select a specifc experiment using a specific model. These experiement files end on "_exp.py".
Hyperparamaters are selected from the papers we are trying to replicate.

possible MODEL values are:

- lstm_crf
- lstm_cnn_crf
- jpnn
- lstm_dist
- lstm_er

possible LEVEL values are a bit specific to the various models:

- lstm_crf
    - [sentence, paragraph, document]
- lstm_cnn_crf
    - [paragraph, document]
- jpnn
    - [paragraph, document]
- lstm_dist
    - [paragraph, document]
- lstm_er
    - [paragraph, document]   


different modes are:

    train, test or both (does both train and test)


for GPU you simply pass the device id of the GPU you want to use. NOTE! lstm_er will currently break on GPU due to Deep Graph Library..



## Experiment 1: Replication

TBA


### Segmentation Model

    - lstm_crf is a model aimed to solve segmentation, i.e. task 1 (seg)


### Models on Segment Level

    - jpnn is aims to solve task 2 and 3

    - lstm_dist aim to solve task 1,2 and 3

### End-to-End models

    - lstm_cnn_crf aim to solve all tasks

    - lstm_er aim to solve all tasks


## Experiment 2: Extending


    To understand how jpnn and lstm_dist perform given that we instead of giving these models gold segments, we give them
    predicted segments, i.e. output of the lstm_crf.

    We do this by creating a new model which is lstm_crf + jpnn and lstm_crf + lstm_dist. i.e. we add the layers (the whole token_module) of the lstm_crf to the jpnn and lstm_dist. Then we can load in the pretrained weights for lstm_crf into the lstm_crf part of the new model and load the weights of jpnn/lstm-dist into the jpnn/lstm-dist part. hence, we get a model
    which can all task (almsot for jpnn).


    TBA