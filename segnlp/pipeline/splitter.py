
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


        def premade_split(indexes, premade_splits:dict=None):

            if premade_splits is not None:
                train = [i for i in indexes if i in premade_splits["train"]]
                test = [i for i in indexes if i in premade_splits["test"]]
            else:
                train, test = train_test_split(idxs, test_size=0.33, shuffle=True)
        
            train, val  = train_test_split(train, test_size=0.1, shuffle=True)
            return {0:{
                        "train": train,
                        "val": val,
                        "test":test
                        }
                    }    


        def new_split(idxs):
            train, test  = train_test_split(idxs, test_size=0.3, shuffle=True)
            train, val  = train_test_split(train, test_size=0.1, shuffle=True)
            return {0:{
                        "train": train,
                        "val": val,
                        "test":test
                        }
                    }


        def cv_split(idxs):
            kf = KFold(n_splits=10, shuffle=True)
            splits = {i:{"train": train_index,  "val":test_index} for i, (train_index, test_index) in enumerate(kf.split(idxs))}
            return splits


        if os.path.exists(self._path_to_splits):
            return

        idxs = np.arange(self._n_samples)

        if self.evaluation_method == "cross_validation":
            splits = cv_split(idxs)

        elif self.sample_level != self.dataset_level:
            splits = new_split(idxs)

        else:
            splits = premade_split(idxs, premade_splits = premade_splits)


        with open(self._path_to_splits, "wb") as f:
            pkl.dump(splits, f)