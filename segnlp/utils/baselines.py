
# basics
import random
import numpy as np


# pytroch
import torch

# segnlp
from .bio_decoder import BIODecoder
from .array import ensure_numpy


class RandomBaseline:


    def __init__(self, task_labels: dict, random_seed:int, weights : dict = None):
        random.seed(random_seed)
        self._task_labels = task_labels
        self._tasks = tuple(list(self._task_labels.keys()))
        self._clfs  = {}
        

        for task, labels in task_labels.items():

            if weights is None:
                label_weights = None
            else:
                label_weights = weights[task]

            
            if task == "link":
                self._clfs[task]  = self.__create_link_clf()
            else:
                self._clfs[task]  = self.__create_random_clf(labels = labels, weights = label_weights)


    def __create_link_clf(self):

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


    def __create_random_clf(
                            self,
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


    def pred(self, df):

        k = len(df)
        df["link"] = None
        for task in self._task_labels:

            if task ==  "link":
            
                for si, sample_df in df.groupby("sample_id", sort = False):
                    n_segs = len(sample_df)
                    values = self._clfs["link"](labels = list(range(n_segs)), k= n_segs)
                    
                    df.loc[df["sample_id"] == si,"link"] = values
                    
            else:
                df[task] = self._clfs[task](k=k)
    
        return df


class SentenceRandomBaseline(RandomBaseline):


    def __init__(self, task_labels: dict,  random_seed:int, p:float, weights : dict = None):
        super().__init__(
                        task_labels = task_labels, 
                        random_seed = random_seed, 
                        weights = weights
                        )
        self._p = p
        self._bio_decoder = BIODecoder()


    def pred(self, df):

        df_size = len(df)
        o_index = df.index
        
        # # we randomly predict BIO segs
        # segs = self._clfs["seg"](k=df_size)

        # # we decode segments from the randomly segmented segemnts
        # # we get the sample start indexes from sample lengths. We need this to tell de decoder where samples start
        # sample_sizes = df.groupby("sample_id", sort = False).size().to_numpy()
        # sample_end_idxs = np.cumsum(sample_sizes)
        # sample_start_idxs = np.concatenate((np.zeros(1), sample_end_idxs))[:-1]
        
        # seg_ids = self._bio_decoder(segs, sample_start_idxs = sample_start_idxs.astype(int))


        # We set sentences as segments ids
        df["seg_id"] = None

        # then we selected sentences 
        n_sentences = df["sentence_id"].nunique()

        # then we extract the length of each sentence
        sentence_lengths = df.groupby("sentence_id", sort = False).size().to_numpy()

        # then we set setting sentence_id to  index so we can filter easily
        df.set_index(["sentence_id"], inplace = True)

        # the we select sentences
        mask = ensure_numpy(torch.zeros(n_sentences).bernoulli_(self._p)).astype(bool)
        selected_sentences = np.arange(n_sentences)[mask]

        # then we select rows using our selected sentences and add a new range of ids 
        # as segment ids.
        df.loc[selected_sentences, "seg_id"] = np.repeat(
                                                        np.arange(len(selected_sentences)),
                                                        sentence_lengths[selected_sentences]
                                                        )

        # We create a dataframe where rows are segments instead of tokens
        seg_df = df.groupby("seg_id", sort = False).first()
        
        # we pass the seg_dataframe to the RandomBaseline.pred to predict labels for each segment
        seg_df = super().pred(seg_df)
    
        # set index to seg_id expand rows easier
        df = df.set_index(["seg_id"])
        df["seg_id"] = df.index
        
        for seg_id, row in seg_df.iterrows():
            task_preds = [row[t] for t in self._tasks]
            df.loc[seg_id, self._tasks] = task_preds
        
        df.index = o_index
        return df


class MajorityBaseline:

    def __new__(self, task_label:dict,  random_seed:int):
        task_label = {t:[l] for t,l in task_label.items()}

        return RandomBaseline(
                        task_labels = task_label, 
                        random_seed = random_seed
                        )


class SentenceMajorityBaseline:

    def __new__(self, task_label:dict,  random_seed:int, p: float):
        task_label = {t:[l] for t,l in task_label.items()}

        return SentenceRandomBaseline(
                            task_labels = task_label, 
                            random_seed = random_seed, 
                            p = p
                            )