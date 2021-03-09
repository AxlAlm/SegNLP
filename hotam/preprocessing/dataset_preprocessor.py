


#basics
from tqdm import tqdm
import numpy as np
from typing import Union, Sequence
import os
import json
import pickle
from collections import Counter
import pandas as pd
from IPython.display import display

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
from hotam.datasets import get_dataset


#sklearn
from sklearn.model_selection import train_test_split

class PreProcessedDataset(ptl.LightningDataModule):

    def __init__(self, name:str, dir_path:str, label_encoders:dict):
        self._name = name
        print(name)
        self.label_colors = get_dataset(name).label_colors()
        self.label_encoders = label_encoders
        self._fp = os.path.join(dir_path, f"{name}_data.hdf5")
        self.data = h5py.File(self._fp, "r")
        self._size = self.data["ids"].shape[0]
        self.prediction_level = "token"

        self._stats = pd.read_csv(os.path.join(dir_path, f"{name}_stats.csv"))
        self._stats.columns = ["split_id", "split", "task", "label", "count"]

        with open(os.path.join(dir_path, f"{name}_splits.pkl"), "rb") as f:
            self._splits = pickle.load(f)
        

    def __getitem__(self, key:Union[np.ndarray, list]) -> ModelInput:
        Input = ModelInput(
                            label_encoders=self.label_encoders, 
                            label_colors=self.label_colors
                            )

        sorted_key = np.sort(key)
        lengths = self.data[self.prediction_level]["lengths"][sorted_key]
        max_len = max(lengths)
        lengths_decending = np.argsort(lengths)[::-1]

        Input._ids = self.data["ids"][sorted_key][lengths_decending]
        
        for group in self.data:

            if group == "ids":
                continue
            
            Input[group] = {}
            for k, v in self.data[group].items():
                
                a = v[sorted_key]
            
                if len(a.shape) > 1:
                    a = a[:, :max_len]
            
                Input[group][k] = a[lengths_decending]
            
        Input.to_tensor()
        return Input
    

    def __len__(self):
        return self._size


    @property
    def name(self):
        return self._name

    @property
    def info(self):
        
        structure = { }
        for group in self.data.keys():

            if group == "ids":
                structure["ids"] = f'dtype={str(self.data["ids"].dtype)}, shape={self.data["ids"].shape}'
                continue

            structure[group] = {}
            for k, v in self.data[group].items():
                structure[group][k] = f"dtype={str(v.dtype)}, shape={v.shape}"


        s = f"""
            Structure:        
            
            {json.dumps(structure, indent=4)}

            Size            = {self._size}
            File Size (MBs) = {str(round(os.path.getsize(self._fp) / (1024 * 1024), 3))}
            Filepath       = {self._fp}
            """
        print(s)

    @property
    def stats(self):
        return self._stats

    @property
    def splits(self):
        return self._splits


    def train_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["train"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0], num_workers=8)


    def val_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["val"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0], num_workers=8) #, shuffle=True)


    def test_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["test"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0], num_workers=8)



class DataPreprocessor:

    def _init_DataPreprocessor(self):
        self.__init_storage_done = False


    def __setup_h5py(self, file_path:str):
        self.h5py_f = h5py.File(file_path, 'w')


    def __init_store(self, Input:ModelInput):
        
        self.h5py_f.create_dataset("ids", data=Input.ids, dtype=np.int16, chunks=True, maxshape=(None,))

        for level in Input.levels:
            
            for k,v in Input[level].items():
                v = ensure_numpy(v)
                #dynamic_shape = tuple([None for v in enumerate(v.shape)])
                #self.h5py_f.create_dataset(k, dynamic_shape, dtype=v.dtype)
                max_shape = tuple([None for v in enumerate(v.shape)])

                name = f"/{level}/{k}"
                if "<U" in str(v.dtype):
                    self.h5py_f.create_dataset(name, data=v.tolist(), chunks=True, maxshape=max_shape)
                else:
                    self.h5py_f.create_dataset(name, data=v, dtype=v.dtype, chunks=True, maxshape=max_shape)

        self.__init_storage_done = True
    

    def __append_store(self, Input:ModelInput):

        last_sample_i = self.h5py_f["ids"].shape[0]
        self.h5py_f["ids"].resize((self.h5py_f["ids"].shape[0] + Input.ids.shape[0],))
        self.h5py_f["ids"][last_sample_i:] = Input.ids

        for level in Input.levels:
            for k,v in Input[level].items():
                v = ensure_numpy(v)
                k = f"/{level}/{k}"

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


    def __set_splits(self, dump_dir:str, dataset:DataSet):


        def create_new_splits(ids):
            train, test  = train_test_split(ids,test_size=0.3)
            train, val  = train_test_split(train,test_size=0.1)
            return {0:{
                        "train": train,
                        "val": val,
                        "test":test
                        }
                    }

        splits = dataset.splits

        if self.sample_level != dataset.level:
            splits = create_new_splits(self.h5py_f["ids"][:])

        file_path = os.path.join(dump_dir, f"{dataset.name}_splits.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(splits, f)

        return splits


    def __calc_stats(self, dump_dir:str, splits:dict, dataset_name:str):
        
        collected_counts = {split_id:{split:{t:[] for t in self.all_tasks} for split in ["train", "val", "test"]} for split_id in splits.keys()}
        
        lengths = self.h5py_f[self.prediction_level]["lengths"][:]
        span_lengths = self.h5py_f["span"]["lengths_tok"][:]
        lengths_span = self.h5py_f["span"]["lengths"][:]
        non_spans_mask = self.h5py_f["span"]["none_span_mask"][:]

        ids  = self.h5py_f["ids"]

        for task in self.all_tasks:
            encoded_labels = self.h5py_f[self.prediction_level][task][:]

            sample_iter = enumerate(zip(ids, lengths, encoded_labels))
            for i, (ID, length, labels) in sample_iter:
                
                if task == "link" and self.prediction_level == "token":

                    #we have the span labels so just use them and expand them?
                    stl = span_lengths[i][:lengths_span[i]]
                    ns = non_spans_mask[i][:lengths_span[i]]
                    decoded_labels = self.decode_token_links(
                                                            labels[:length].tolist(), 
                                                            span_token_lengths = stl, 
                                                            none_spans = ns
                                                            )
                else:
                    decoded_labels = self.decode_list(labels[:length].tolist(), task)

                labe_counts = dict(Counter(decoded_labels))

                for split_id, splits_dict in splits.items():
                    counts = labe_counts.copy()
         
                    for split, split_ids in splits_dict.items():
                        if ID in split_ids:
                            collected_counts[split_id][split][task].append(counts)


        split_id_dfs = []
        for split_id, sub_dict in collected_counts.items():
            
            split_dfs = []
            for split, task_counts in sub_dict.items():
                
                task_dfs = [] 
                for task, counts in task_counts.items():
                    task_dfs.append(pd.DataFrame(counts).sum().T)

                split_dfs.append(pd.concat(task_dfs, keys=self.all_tasks))
            
            split_id_dfs.append(pd.concat(split_dfs, keys=["train", "val", "test"]))
        
        df = pd.concat(split_id_dfs, keys=list(splits.keys()))
        #stats = df.to_dict()
   
        file_path = os.path.join(dump_dir, f"{dataset_name}_stats.csv")
        df.to_csv(file_path)
        # with open(file_path,"wb") as f:
        #     pickle.dump(stats, f)


    def process_dataset(self, dataset:DataSet, chunks:int = 50, dump_dir:str = None) -> PreProcessedDataset:
        
        file_path = os.path.join(dump_dir, f"{dataset.name}_data.hdf5")
        self.__setup_h5py(file_path=file_path) 


        progress_bar = tqdm(total=len(dataset), desc="Processing and Storing Dataset")
        #last_id = 0
        for i in range(0, len(dataset), chunks):
            Input = self(dataset[i:i+chunks])

            # Input._ids = Input._ids + (last_id + (1 if last_id else 0))
            # last_id = Input.ids[-1]

            if not self.__init_storage_done:
                self.__init_store(Input)
            else:
                self.__append_store(Input)

            progress_bar.update(chunks)

        progress_bar.close()

        splits = self.__set_splits(dump_dir, dataset=dataset)
        self.__calc_stats(dump_dir, splits, dataset.name)

        self.h5py_f.close()

        return PreProcessedDataset(name=dataset.name, dir_path=dump_dir, label_encoders=self.encoders)



        # if self._data_is_stored:
        #     out = self.__get_stored_data(key)
        # else:
        # #     if self.__processor is not None:
        # #         out = self.preprocessor(self.data[key])
        # #     else:
        #     out = self.data[key]

        # return out
