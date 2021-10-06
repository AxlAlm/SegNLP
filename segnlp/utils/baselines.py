
# basics
import random
import numpy as np


# segnlp
from .bio_decoder import BIODecoder


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
                self._clfs[task]  = self.__create_link_clf(labels = labels)
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


    def __init__(self, task_labels: dict,  random_seed:int, weights : dict = None):
        super().__init__(
                        task_labels = task_labels, 
                        random_seed = random_seed, 
                        weights = weights
                        )
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

        # # add segment ids
        df["seg_id"] = df["sentence_id"]        

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


class MajorityLabelBaseline:

    def __new__(self, majority_labels:dict,  random_seed:int ):
        RandomBaseline(task_labels = majority_labels, random_seed = random_seed)


class SentenceMajorityLabelBaseline:

    def __new__(self, majority_labels:dict,  random_seed:int ):
        SentenceRandomBaseline(task_labels = majority_labels, random_seed = random_seed)