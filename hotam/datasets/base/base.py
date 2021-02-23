

import pytorch_lightning as ptl

#pytroch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler
import torch

#h5py
import h5py


class DataSet(ptl.LightningDataModule):


    def __init__(self, splits:dict, Preprocessor:Preprocessor):
        
        self._data_is_stored = False
        self._data_is_processed = False


    def __getitem__(self, key):

        if self._data_is_stored:
            data = self._h5py_storage[key]

        elif self._data_is_processed:
            data = self._preprocessed_data[k]

        else:
            data = Preprocessor(self.data[key])

        return data


    def __setup_h5py(self, file_name):
        self.h5py_f = h5py.File(f'/tmp/{file_name}.hdf5', 'w')


    def process(self, chunks=50, prefix:str=""):
        
        if self._store_data:
            self.__setup_h5py(file_name)


        for i in range(0, len(self), chunks):
            Input = Preprocessor(self.data[i:i+chunks])

            if self._store_data:
                self.store(Input)
            else:



    def store(self, Input:Input):

        grp.create_dataset("another_dataset", (50,), dtype='f')
        grp.create_dataset("another_dataset", (50,), dtype='f')


    def train_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["train"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0]) #, shuffle=True)


    def val_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["val"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0]) #, shuffle=True)


    def test_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["test"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0])
