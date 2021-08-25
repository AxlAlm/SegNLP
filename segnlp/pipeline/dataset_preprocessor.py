


#basics
from tqdm import tqdm
import numpy as np
from typing import Union, Sequence
import os
from collections import Counter
import pandas as pd
import shutil

#h5py
import h5py

# segnlp
from segnlp.datasets.base import DataSet
from segnlp.utils import Input
import segnlp.utils as utils
from segnlp import get_logger


logger = get_logger("DatasetPreprocessor")


class DatasetPreprocessor:


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


    def _preprocess_dataset(self, dataset:DataSet):
        
        if os.path.exists(self._path_to_preprocessed_data):
            return None

        self.__setup_h5py(file_path=self._path_to_preprocessed_data) 

        self._nr_sample = 0
        for i in tqdm(range(len(dataset)), desc="Processing and Storing Dataset"):
            input = self._process_text(dataset[i])

            if not self.__init_storage_done:
                self.__init_store(input)
            else:
                self.__append_store(input)

            self._nr_sample += len(input)

        self.h5py_f.close()
        



    # def _preprocess_dataset(self, dataset:DataSet):

    #     if os.path.exists(self._path_to_preprocessed_data):

    #         logger.info(f"Loading preprocessed data from {self._path_to_preprocessed_data}")
    #         return self.__preprocess_dataset(dataset)

    #     except OSError as e:
    #         logger.info(f"Loading failed. Will continue to preprocess data")
    #         try:
    #             shutil.rmtree(dump_dir)
    #         except FileNotFoundError as e:
    #             pass

        # try:
        #     return self._preprocess_dataset(dataset)

        # except BaseException as e:
        #     shutil.rmtree(dump_dir)
        #     raise e






    # def __calc_stats(self, dump_dir:str, splits:dict, dataset_name:str):
        
    #     collected_counts = {split_id:{split:{t:[] for t in self.all_tasks} for split in ["train", "val", "test"]} for split_id in splits.keys()}
        
    #     lengths = self.h5py_f[self.prediction_level]["lengths"][:]
    #     span_lengths = self.h5py_f["span"]["lengths_tok"][:]
    #     lengths_span = self.h5py_f["span"]["lengths"][:]
    #     non_spans_mask = self.h5py_f["span"]["none_span_mask"][:]

    #     idxs  = self.h5py_f["ids"]

    #     for task in self.all_tasks:
    #         encoded_labels = self.h5py_f[self.prediction_level][task][:]

    #         sample_iter = enumerate(zip(idxs, lengths, encoded_labels))
    #         for i, (ID, length, labels) in sample_iter:
                
    #             if task == "link" and self.prediction_level == "token":

    #                 #we have the span labels so just use them and expand them?
    #                 stl = span_lengths[i][:lengths_span[i]]
    #                 ns = non_spans_mask[i][:lengths_span[i]]
    #                 decoded_labels = self.decode_token_links(
    #                                                         labels[:length].tolist(), 
    #                                                         span_token_lengths = stl, 
    #                                                         none_spans = ns
    #                                                         )
    #             else:
    #                 decoded_labels = self.decode_list(labels[:length].tolist(), task)

    #             labe_counts = dict(Counter(decoded_labels))

    #             for split_id, splits_dict in splits.items():
    #                 counts = labe_counts.copy()
         
    #                 for split, split_idxs in splits_dict.items():
    #                     if ID in split_idxs:
    #                         collected_counts[split_id][split][task].append(counts)


    #     split_id_dfs = []
    #     for split_id, sub_dict in collected_counts.items():
            
    #         split_dfs = []
    #         for split, task_counts in sub_dict.items():
                
    #             task_dfs = [] 
    #             for task, counts in task_counts.items():
    #                 task_dfs.append(pd.DataFrame(counts).sum().T)

    #             split_dfs.append(pd.concat(task_dfs, keys=self.all_tasks))
            
    #         split_id_dfs.append(pd.concat(split_dfs, keys=["train", "val", "test"]))
        
    #     df = pd.concat(split_id_dfs, keys=list(splits.keys()))
    #     #stats = df.to_dict()
   
    #     file_path = os.path.join(dump_dir, f"{dataset_name}_stats.csv")
    #     df.to_csv(file_path)
    #     # with open(file_path,"wb") as f:
    #     #     pickle.dump(stats, f)

