


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
import multiprocessing
import shutil

#pytroch lighnting
import pytorch_lightning as ptl

#pytroch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler
import torch

#h5py
import h5py

# segnlp
from segnlp.datasets.base import DataSet
from segnlp.utils import Input
from segnlp.utils import DataModule
import segnlp.utils as utils
from segnlp import get_logger

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

logger = get_logger("DATA-PREPROCESSING")


class DataPreprocessor:

    def _init_DataPreprocessor(self):
        self.__init_storage_done = False


    def __setup_h5py(self, file_path:str):
        self.h5py_f = h5py.File(file_path, 'w')


    def __init_store(self, input:Input):
        
        self.h5py_f.create_dataset("ids", data=input.ids, dtype=np.int16, chunks=True, maxshape=(None,))

        for level in input.levels:
            
            for k,v in input[level].items():
                v = utils.ensure_numpy(v)
                #dynamic_shape = tuple([None for v in enumerate(v.shape)])
                #self.h5py_f.create_dataset(k, dynamic_shape, dtype=v.dtype)
                max_shape = tuple([None for v in enumerate(v.shape)])

                name = f"/{level}/{k}"
                if "<U" in str(v.dtype):
                    self.h5py_f.create_dataset(name, data=v.tolist(), chunks=True, maxshape=max_shape)
                else:

                    fillvalue = 0
                    if k in self.all_tasks:
                        fillvalue = -1

                    self.h5py_f.create_dataset(name, data=v, dtype=v.dtype, chunks=True, maxshape=max_shape, fillvalue=fillvalue)

        self.__init_storage_done = True
    

    def __append_store(self, input:Input):

        last_sample_i = self.h5py_f["ids"].shape[0]
        self.h5py_f["ids"].resize((self.h5py_f["ids"].shape[0] + input.ids.shape[0],))
        self.h5py_f["ids"][last_sample_i:] = input.ids

        for level in input.levels:
            for k,v in input[level].items():
                v = utils.ensure_numpy(v)
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


    def __set_splits(self, dump_dir:str, dataset:DataSet, size:int, evaluation_method:str):

        
        def split(size, split_idx):
            ids = np.arange(size)
            train = ids[:split_idx]
            test = ids[split_idx:]
            train, val  = train_test_split(train,test_size=0.1, shuffle=False)
            return {0:{
                        "train": train,
                        "val": val,
                        "test":test
                        }
                    }    

        def split_new(size):
            ids = np.arange(size)
            train, test  = train_test_split(ids, test_size=0.3, shuffle=False)
            train, val  = train_test_split(train, test_size=0.1, shuffle=False)
            return {0:{
                        "train": train,
                        "val": val,
                        "test":test
                        }
                    }

        def cv_split(size):
            ### MTC normaly uses Cross Validation
            kf = KFold(n_splits=10, shuffle=False, random_state=42)
            ids = np.arange(size)
            splits = {i:{"train": train_index,  "val":test_index} for i, (train_index, test_index) in enumerate(kf.split(ids))}
            return splits


        if evaluation_method == "cross_validation":
            splits = cv_split(size)

        elif self.sample_level != dataset.level:
            splits = split_new(size)

        else:
            splits = split(size, dataset.split_idx)

        file_path = os.path.join(dump_dir, f"{dataset.name()}_splits.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(splits, f)

        return splits


    def __calc_stats(self, dump_dir:str, splits:dict, dataset_name:str):
        
        collected_counts = {split_id:{split:{t:[] for t in self.all_tasks} for split in ["train", "val", "test"]} for split_id in splits.keys()}
        
        lengths = self.h5py_f[self.prediction_level]["lengths"][:]
        span_lengths = self.h5py_f["span"]["lengths_tok"][:]
        lengths_span = self.h5py_f["span"]["lengths"][:]
        non_spans_mask = self.h5py_f["span"]["none_span_mask"][:]

        idxs  = self.h5py_f["ids"]

        for task in self.all_tasks:
            encoded_labels = self.h5py_f[self.prediction_level][task][:]

            sample_iter = enumerate(zip(idxs, lengths, encoded_labels))
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
         
                    for split, split_idxs in splits_dict.items():
                        if ID in split_idxs:
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


    def __process_dataset(self, dataset:DataSet, evaluation_method:str, dump_dir:str = None) -> DataModule:
        
        if dataset.name() == "MTC":
            self.am_extraction = "from_list"

        path_to_data = os.path.join(dump_dir, f"{dataset.name()}_data.hdf5")
        self.__setup_h5py(file_path=path_to_data) 

        size = 0
        for i in tqdm(range(len(dataset)), desc="Processing and Storing Dataset"):
            input = self(dataset[i])

            if not self.__init_storage_done:
                self.__init_store(input)
            else:
                self.__append_store(input)

            size += len(input)

        splits = self.__set_splits(dump_dir, dataset=dataset, size=size, evaluation_method=evaluation_method)
        self.__calc_stats(dump_dir, splits, dataset.name())

        self.h5py_f.close()

        return DataModule(
                                name=dataset.name(), 
                                dir_path=dump_dir, 
                                prediction_level=self.prediction_level
                                )


    def process_dataset(self, dataset:DataSet, dump_dir:str, evaluation_method:str):

        path_to_data = os.path.join(dump_dir, f"{dataset.name()}_data.hdf5")

        if os.path.exists(path_to_data):
            try:
                logger.info(f"Loading preprocessed data from {path_to_data}")
                return DataModule(
                                                    name=dataset.name(),
                                                    dir_path=dump_dir,
                                                    prediction_level=dataset.prediction_level
                                                    )
            except OSError as e:
                logger.info(f"Loading failed. Will continue to preprocess data")
                try:
                    shutil.rmtree(dump_dir)
                except FileNotFoundError as e:
                    pass

        try:
            return self.__process_dataset(
                                            dataset, 
                                            evaluation_method=evaluation_method, 
                                            dump_dir=dump_dir
                                            )
        except BaseException as e:
            shutil.rmtree(dump_dir)
            raise e

