


# Dataset


## Dataset

    - Be weary of using dataset which come preprocessed. Although it might be convenient that part of all of the preprocessing is done its still the case that preprocessing can be important and hence something you want to make sure if correct and in controll of.
    Even if we know where the preprocessed data is from and how its created it can still be important to implement those steps yourself to both for replication purposes, inference purposes and for comparison between preprocessing methods; e.g. does Stanza/CoreNLP, SpaCy or just a whitespace tokenizer yield the best result.
    An NLP system should always try to replicate a real-world-context, especiallly for evaluation. I.e. the input to the system should be as raw as possible.
    


## Shuffling

    - shuffle each epoch



# NNs


## Layer Dimensionality 




## Activations



## Optimization and Learning Scheduling



## Weight Averaging

https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
https://arxiv.org/abs/1803.05407



## Training Tips

    - Early Stopping
        - Reduce overfitting

    - Gradient Clipping
        Useful in RNNS
    

    - Learning Rate scheduling


    - Dropout
        - reduce overfitting
        - prevent Features co-adapting ( a feature can only be useful in the presence of a particular other feature??!?)






## Metrics




# FEATURES


## OOV
    

    - Create a vocabulary from a cropus which is not the Dataset itself. E.g. BNC or ...

    - Use Byte-Pair encoding vocabulary instead of token vocabulary


## Word Embeddings


    #### Fine-Tuning

    ####

    https://arxiv.org/pdf/2010.05006.pdf


## Byte Pair embeddings




# Segmentation Encoding Schems / Segment Representation Schemes



https://aclanthology.org/W09-1119.pdf
    
https://aclanthology.org/W15-5603.pdf