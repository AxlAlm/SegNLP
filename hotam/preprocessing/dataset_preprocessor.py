


#basics
from tqdm import tqdm
import numpy as np
from typing import Union, Sequence
import os

#pytroch lighnting
import pytorch_lightning as ptl

#pytroch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler
import torch

#h5py
import h5py

# hotam
from hotam.datasets.base import DataSet
from hotam.nn import ModelInput
from hotam.nn.model_input import ModelInput
from hotam.utils import ensure_numpy


class PreProcessedDataset(ptl.LightningDataModule):

    def __init__(self, h5py_file_path:str,  splits:dict):
        self._fp = h5py_file_path
        self.data = h5py.File(self._fp, "r")
        self.splits = splits
        self._size = self.data["id"].shape[0]


    def __getitem__(self, key:Union[np.ndarray, list]) -> ModelInput:
        Input = ModelInput()
        key = ensure_numpy(key)
        sorted_idx = np.argsort(key)
        original_idx = np.argsort(sorted_idx)
        sorted_key = key[sorted_idx]

        lengths = self.data["lengths_tok"][sorted_key]
        lengths_decending = np.argsort(lengths[::-1])

        for dset in self.data:
            data = self.data[dset][sorted_key]
            Input[dset] =  data[original_idx]
        Input.to_tensor()
        return Input
    

    def __len__(self):
        return self._size


    def info(self):
        s = f"""
            keys            = {[dset for dset in self.data]}
            size            = {self._size}
            file size (MBs) = {str(round(os.path.getsize(self._fp) / (1024 * 1024), 3))}
            file path       = {self._fp}
            """
        print(s)


    def stats(self):
        pass


    def train_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["train"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0])


    def val_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["val"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0]) #, shuffle=True)


    def test_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["test"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0])



class DataPreprocessor:

    def _init_DataPreprocessor(self):
        self.__init_storage_done = False


    def __setup_h5py(self, file_path:str):
        self.h5py_f = h5py.File(file_path, 'w')


    def __init_store(self, Input:ModelInput):

        for k,v in Input.items():
            v = ensure_numpy(v)
            #dynamic_shape = tuple([None for v in enumerate(v.shape)])
            #self.h5py_f.create_dataset(k, dynamic_shape, dtype=v.dtype)
            max_shape = tuple([None for v in enumerate(v.shape)])
            if "<U" in str(v.dtype):
                self.h5py_f.create_dataset(k, data=v.tolist(), chunks=True, maxshape=max_shape)
            else:
                self.h5py_f.create_dataset(k, data=v, dtype=v.dtype, chunks=True, maxshape=max_shape)

        self.__init_storage_done = True
    

    def __append_store(self, Input:ModelInput):

        for k,v in Input.items():
            #try:
            v = ensure_numpy(v)

            last_sample_i = self.h5py_f[k].shape[0]

            nr_dims = len(v.shape)
            if nr_dims == 1:
                new_shape = (self.h5py_f[k].shape[0] + v.shape[0],)
            else:
                nr_rows = self.h5py_f[k].shape[0] + v.shape[0]
                max_shape = np.maximum(self.h5py_f[k].shape[1:], v.shape[1:])
                new_shape = (nr_rows, *max_shape)
        
            self.h5py_f[k].resize(new_shape)

            if nr_dims > 2:
                self.h5py_f[k][last_sample_i:,:v.shape[1], :v.shape[2]] = v
            elif nr_dims == 2:
                self.h5py_f[k][last_sample_i:,:v.shape[1]] = v
            else:
                self.h5py_f[k][last_sample_i:] = v


    def load_preprocessed_dataset(self, file_path):
        return PreProcessedDataset(h5py_file_path)


    def process_dataset(self, dataset:DataSet, chunks:int = 50, dump_dir:str = None) -> PreProcessedDataset:
        
        file_path = os.path.join(dump_dir,"data.hdf5")
        self.__setup_h5py(file_path=file_path) 

        progress_bar = tqdm(total=len(dataset), desc="Processing and Storing Dataset")
        last_id = 0
        for i in range(0, len(dataset), chunks):
            Input = self(dataset[i:i+chunks])

            Input["id"] = Input["id"] + (last_id+1)
            last_id = Input["id"][-1]

            if not self.__init_storage_done:
                self.__init_store(Input)
            else:
                self.__append_store(Input)

            progress_bar.update(chunks)

        progress_bar.close()

        self.h5py_f.close()

        splits = dataset.splits
        if self.sample_level != dataset.level:
            splits = create_new_splits()

        return PreProcessedDataset(file_path, splits=splits)


        # if self._data_is_stored:
        #     out = self.__get_stored_data(key)
        # else:
        # #     if self.__processor is not None:
        # #         out = self.preprocessor(self.data[key])
        # #     else:
        #     out = self.data[key]

        # return out
    # def add_processor(self, processor):
    #     self.__processor = processor
    

    # def __get_stored_data(self, key):

    #     Input = Input()
    #     for dset in h5py_datasets:
    #         Input.add(k, dset[key])

    #     return Input