# HotAM

![](https://github.com/AxlAlm/HotAM/blob/main/HOTAM_LOGO.png)

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

