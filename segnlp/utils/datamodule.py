

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

# pytorch lightning
import pytorch_lightning as ptl

#segnlp
import segnlp
from .batch_input import BatchInput
from segnlp import utils



class DataModule(ptl.LightningDataModule):

    """

    DataModule is an access point or a intermediate gateway to the dataset which is located
    on disk in an H5PY format.

    To access the dataset on disk on can give the DataModule a list of int as indexes.

    """

    def __init__(self, 
                path_to_data:str, 
                batch_size: str,
                split_id : int = 0
                ):

        self._df_fp = os.path.join(path_to_data, "df.hdf5")
        self._pwf_fp = utils.check_file(os.path.join(path_to_data, "pwf.hdf5"))
        self._psf_fp = utils.check_file(os.path.join(path_to_data, "psf.hdf5"))

        with open(os.path.join(path_to_data, f"splits.pkl"), "rb") as f:
            self._splits = pickle.load(f)
        
        self.batch_size = batch_size
        self.split_id = split_id
        

    def __get_df(self, key):
        return pd.read_hdf(self._df_fp, where = f"index in {[str(k) for k in key]}")


    def __get_pwf(self, key):

        if self._pwf_fp is None:
            return

        with h5py.File(self._pwf_fp, "r") as f:
            return np.array([f["word_embs"][i] for i in key])


    def __get_psf(self, key):

        if self._psf_fp is None:
            return

        with h5py.File(self._psf_fp, "r") as f:
            return np.array([f["seg_embs"][i] for i in key])



    def __getitem__(self, key:Union[np.ndarray, list]) -> BatchInput:
        return BatchInput(
                    df = self.__get_df(key),
                    word_embs = self.__get_pwf(key),
                    seg_embs = self.__get_psf(key),
                    batch_size = self.batch_size
                    )


    def __get_dataloader(self, split:str): #split = {"train", "test", "val"}

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
                            collate_fn=lambda x:x[0], 
                            num_workers = segnlp.settings["dl_n_workers"]
                            )

    
    def change_split_id(self, n):
        self.split_id = n


    def train_dataloader(self):
        return self.__get_dataloader("train")


    def val_dataloader(self):
        return self.__get_dataloader("val")


    def test_dataloader(self):
        return self.__get_dataloader("test")