
#basics
import numpy as np
import pickle as pkl
import os

#sklearn 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



# segnlp
from segnlp.datasets.base import DataSet



class Splitter:


    def _set_splits(self,  premade_splits):


        def split(ids, premade_splits:dict=None):

            if premade_splits is not None:
                train = [i for i in ids if i in premade_splits["train"]]
                test = [i for i in ids if i in premade_splits["test"]]
            else:
                train, test = train_test_split(ids, test_size=0.33, shuffle=True)
        
            train, val  = train_test_split(train, test_size=0.1, shuffle=True)
            return {0:{
                        "train": train,
                        "val": val,
                        "test":test
                        }
                    }    

        def split_new(ids):
            train, test  = train_test_split(ids, test_size=0.3, shuffle=True)
            train, val  = train_test_split(train, test_size=0.1, shuffle=True)
            return {0:{
                        "train": train,
                        "val": val,
                        "test":test
                        }
                    }

        def cv_split(ids):
            kf = KFold(n_splits=10, shuffle=True)
            splits = {i:{"train": train_index,  "val":test_index} for i, (train_index, test_index) in enumerate(kf.split(ids))}
            return splits


        if os.path.exists(self._path_to_splits):
            return

        ids = np.arange(self._n_samples)

        if self.evaluation_method == "cross_validation":
            splits = cv_split(ids)

        elif self.sample_level != self.dataset_level:
            splits = split_new(ids)

        else:
            splits = split(ids, premade_splits = premade_splits)


        with open(self._path_to_splits, "wb") as f:
            pkl.dump(splits, f)