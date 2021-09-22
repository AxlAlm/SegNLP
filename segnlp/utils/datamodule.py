

#basics
from typing import Union, List, Tuple
import numpy as np
import pickle
from numpy.lib import utils
import pandas as pd
import os

# h5py
import h5py

# pytorch
import torch
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader


#segnlp
import segnlp
from .batch import Batch
from segnlp import utils
from .label_encoder import LabelEncoder


class DataModule:

    """

    DataModule is an access point or a intermediate gateway to the dataset which is located
    on disk in an H5PY format.

    To access the dataset on disk on can give the DataModule a list of int as indexes.

    """

    def __init__(self, 
                path_to_data : str, 
                batch_size : str,
                label_encoder : LabelEncoder,
                cv : int = 0,
                ):

        self._df_fp : str = os.path.join(path_to_data, "df.hdf5")
        self._pwf_fp : str = utils.check_file(os.path.join(path_to_data, "pwf.hdf5"))
        self._psf_fp : str = utils.check_file(os.path.join(path_to_data, "psf.hdf5"))

        with open(os.path.join(path_to_data, f"splits.pkl"), "rb") as f:
            self._splits : dict = pickle.load(f)
        
        self.batch_size : int = batch_size
        self.split_id : int = cv
        self.label_encoder : LabelEncoder = label_encoder
        

    def __getitem__(self, key:Union[np.ndarray, list]) -> Batch:
        return Batch(
                    df = self.__get_df(key),
                    label_encoder = self.label_encoder,
                    pretrained_features = self.__get_pretrained_features(key),
                    )


    def __get_df(self, key):
        return pd.read_hdf(self._df_fp, mode = "r", where = f"index in {[k for k in key]}")


    def __get_pretrained_features(self, key):

        pretrained_features = {}

        if self._pwf_fp is not None:
        
            with h5py.File(self._pwf_fp, "r") as f:
                pretrained_features["word_embs"] = np.array([f["word_embs"][i] for i in key])


        if self._psf_fp is not None:

            with h5py.File(self._psf_fp, "r") as f:
                pretrained_features["seg_embs"] =  np.array([f["seg_embs"][i] for i in key])

        return pretrained_features


    def __collate_fn(self, x):
        return x[0]


    def step(self, split:str): #split = {"train", "test", "val"}

        # we get the ids for the split, these are the indexes we use to 
        # retrieve the sample for the h5py file
        split_ids = self._splits[self.split_id][split]

        # we shuffle the splits
        np.random.shuffle(split_ids)

        # we create a sampler which splits the split_ids into batches
        # and returns list of indexes
        batch_sampler = BatchSampler(
                                split_ids, 
                                batch_size=self.batch_size, 
                                drop_last=False
                                )


        # The DataLoader take as the dataset the create DataModule object, i.e. self.
        # the DataModule.__getitem__() will be used to retrieve the data from h5py.
        # To sample the batches we use our sampler.

        # I.e. the DataModule is a configurable access point to a dataset on disk; a H5PY dataset.
        # We acces the data by passing a list of indexes to the DataModules which fetches the data.

        #ids are given as a nested list from sampler (e.g [[42, 43]]) 
        # hence using lambda x:x[0] to select the inner list.
        return DataLoader( 
                            self,
                            sampler=batch_sampler,
                            collate_fn=self.__collate_fn, 
                            num_workers = segnlp.settings["dl_n_workers"],
                            )
