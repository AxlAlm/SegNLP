
# Datasets


##### Supported datasets


##### picking a datasets

```python
from hotam.datasets import PE

pe = PE()
```


##### Setting up datasets for experimeting


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

**tasks** decide which task you will test during the experiment. Above we only test "seg" (segmentation of argument components) For complexed task, e.g. combinations of two or more task into one, you add a "_" between. E.g. if we want to combine "seg" and "ac" (argument component classification) we use the task ["seg_ac"]. If we want to do "seg" and "ac" seperately we pass ["seg", "ac"].  Note that the tasks allowed here are decided by the dataset.

**sample_level** decide which level the samples in your experiment will be on. If you set sample level documents, you will be passing documents. If you set sentences you will pass sentence and so on.

**prediction_level** decides which level your model is to be evaluated and expect predictions on. Here we only have two options  "token" and "ac". Passing "token" will make the framework expect predictions for a sample to be on token level and transform samples, encodings and features to match this. For the above configuration our e.g. input embeddings for a sample to the model will be in the following dimension (batch_size, max_nr_tokens). 
If we pass "ac" our framwork will expect us to predict on Argument Components. If we for example change the above configuare from "token" to "ac" our input e.g. embeddings for our model will be in the following shape (batch_size, max_nr_ac_in_doc, max_token_in_acs). Note that some combinations of tasks, sample_level and prediction_level are not allowed as they dont make sense, e.g. task=["seg"] and prediction_level=["ac"] -- segmentation is already given as we are given Argument Components.

**encodings** decides which encodings will be done. If you wish to pass encodings such as characters, words, pos or dephead to your model you decide this here. Encodings are unique IDS.
Note that labels are always encoded

**features** decided the features that will be extracted from the dataset. In above example glove embeddings will be extracted. Multiple features can be passed. All input to features should be a hotam.FeatureModel (read more about features).
