
# basics
import random
from typing import Callable, Dict
import numpy as np

# pytroch
import torch

# segnlp
from .array import ensure_numpy


def random_link_clf():

    """ 
    for link classification we do not select labels from a fixed distribution 
    but instead we set labels to the number of possible segments in a sample.
    I.e. we only predict a random link out of all the possible link paths in a sample.
    """

    def clf(labels, k:int):
        return random.choices(
                population = labels,
                k = k
                )


    return clf


def random_clf(
                labels:list,
                weights: list = None,
                ):

    if weights is None:
        weights = [1 / len(labels)] * len(labels)

    def clf(k:int):
        return random.choices(
                population = labels,
                weights = weights,
                k = k
                )

    return clf


def random_majority_link_clf():

    """ 
    for link classification we do not select labels from a fixed distribution 
    but instead we set labels to the number of possible segments in a sample.
    I.e. we only predict a random link out of all the possible link paths in a sample.
    """

    def clf(labels, k:int):
        
        ##only to self
        #return np.arange(k)

        # only one forward
        #return [min(i+1, k-1) for i in range(k)]

        # only one back
        return [max(0, i-1) for i in range(k)]

        ## link to the segment behind or the one ahead.
        ##  If i == k-1, we take i or i-1. if i == 0, we take i or i+1
        #return [random.choice([max(0, i-1), i, min(i+1, k-1)]) for i in range(k)]
        
    return clf


def majority_clf(label):

    def clf(k:int):
        return [label] * k

    return clf


def create_random_baseline_clfs(task_labels, weights):

    clfs = {}

    for task, labels in task_labels.items():

        if weights is None:
            label_weights = None
        else:
            label_weights = weights[task]

        
        if task == "link":
            clfs[task]  = random_link_clf()
        else:
            clfs[task]  = random_clf(labels = labels, weights = label_weights)

    return clfs


def create_majority_baseline_clfs(task_labels):

    clfs = {}
    for task, label in task_labels.items():
        
        if task == "link":
            clfs[task]  = random_majority_link_clf()
        else:
            clfs[task]  = majority_clf(label = label)
    return clfs


class Baseline:

    def __init__(self, clfs : Dict[str, Callable]):
        self._clfs = clfs

    def __call__(self, df):

        df["seg_id"] = df.index.to_numpy()
        
        k = len(df)
        for task, clf in self._clfs.items():
            df["task"] = None

            if task ==  "link":
                df["target_id"] = None
            
                for si, sample_df in df.groupby("sample_id", sort = False):

                    n_segs = len(sample_df)
                    links = clf(
                                labels = list(range(n_segs)),
                                k = n_segs
                                )
                
                    df.loc[df["sample_id"] == si,"link"] = links
                
                    # We also add target_id
                    seg_ids = sample_df["seg_id"].to_numpy()
                    df.loc[df["sample_id"] == si, "target_id"] = seg_ids[links]

            else:
                df[task] = clf(k=k)


        return df


class SentenceBaseline(Baseline):

    def __init__(self, clfs : Dict[str, Callable], p:float):
        super().__init__(clfs = clfs)
        self._tasks = list(clfs.keys())
        self._p = p
        

    def __call__(self, df):

        o_index = df.index

        # We set sentences as segments ids
        df["seg_id"] = None

        # then we selected sentences 
        n_sentences = df["sentence_id"].nunique()

        # then we extract the length of each sentence
        sent_groups = df.groupby("sentence_id", sort = False)
        sentence_lengths = sent_groups.size().to_numpy()
        sentences_ids = sent_groups.first().index.to_numpy()

        # then we set setting sentence_id to  index so we can filter easily
        df.set_index(["sentence_id"], inplace = True)

        # the we select sentences
        mask = ensure_numpy(torch.zeros(n_sentences).bernoulli_(self._p)).astype(bool)
        sentence_idx = np.arange(n_sentences)[mask]
        sentences_ids = sentences_ids[mask]
        n_selected = len(sentences_ids)

        # then we select rows using our selected sentences and add a new range of ids 
        # as segment ids.
        df.loc[sentences_ids, "seg_id"] = np.repeat(
                                                        np.arange(n_selected),
                                                        sentence_lengths[sentence_idx]
                                                        )

        # We create a dataframe where rows are segments instead of tokens
        seg_df = df.groupby("seg_id", sort = False).first()
        
        # we pass the seg_dataframe to the RandomBaseline.pred to predict labels for each segment
        seg_df = super().__call__(seg_df)
    
        # set index to seg_id expand rows easier
        df = df.set_index(["seg_id"])
        df["seg_id"] = df.index
        
        for seg_id, row in seg_df.iterrows():
            task_preds = [row[t] for t in self._tasks] + [row["target_id"]]
            df.loc[seg_id, self._tasks+ ["target_id"]] = task_preds
        
        df.index = o_index
        return df


class RandomBaseline:

    def __new__(self, task_labels: dict, random_seed:int, weights : dict = None):
        random.seed(random_seed)
        clfs = create_random_baseline_clfs(task_labels, weights)
        return Baseline(clfs = clfs)


class MajorityBaseline:

    def __new__(self, task_labels: dict, random_seed:int):
        random.seed(random_seed)
        clfs = create_majority_baseline_clfs(task_labels)
        return Baseline(clfs = clfs)


class SentenceRandomBaseline(RandomBaseline):

    def __new__(self, task_labels: dict,  random_seed:int, p: float, weights:dict =  None):
        random.seed(random_seed)
        clfs = create_random_baseline_clfs(task_labels, weights)
        return SentenceBaseline(clfs = clfs, p = p)


class SentenceMajorityBaseline(MajorityBaseline):

    def __new__(self, task_labels: dict,  random_seed:int, p: float):
        random.seed(random_seed)
        clfs = create_majority_baseline_clfs(task_labels)
        return SentenceBaseline(clfs = clfs, p = p)

