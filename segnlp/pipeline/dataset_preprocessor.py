


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
import segnlp.utils as utils
from segnlp import get_logger
from segnlp.utils import H5PY_STORAGE

logger = get_logger("DatasetPreprocessor")



class DatasetPreprocessor:


    def _check_data(self):
        return all([
                    os.path.exists(self._path_to_df),
                    True if not self._use_psf else os.path.exists(self._path_to_psf),
                    True if not self._use_pwf else os.path.exists(self._path_to_psf)
                    ])


    def _preprocess_dataset(self, dataset:DataSet):
    
        if self._check_data():
            return None

        self._df_storage = pd.HDFStore(self._path_to_df)

        if self._use_pwf:
            self._pwf_storage = H5PY_STORAGE(
                                                name = "word_embs", 
                                                fp = self._path_to_pwf, 
                                                n_dims = 3, 
                                                dtype =  np.float64, 
                                                )

        if self._use_psf:
            self._psf_storage = H5PY_STORAGE(
                                                name = "seg_embs", 
                                                fp = self._path_to_psf, 
                                                n_dims = 3, 
                                                dtype =  np.float64, 
                                                )

        try:
            self._n_samples = 0
            for i in tqdm(range(len(dataset)), desc="Processing and Storing Dataset"):
                self._process_and_store_samples(doc = dataset[i])

        #for keyboard interrupt
        except BaseException as e:
            shutil.rmtree(self._path_to_data)
            raise e

        self._pwf_storage.close()
        self._psf_storage.close()
        self._df_storage.close()

        

    def _process_and_store_samples(self, doc:dict):

        span_labels = doc.get("span_labels", None)
        doc = doc["text"]   

        doc_df = self._process_text(doc)

        if self.input_level != self.sample_level:
            samples = doc_df.groupby(f"{self.sample_level}_id", sort=False)
        else:
            samples = [(None, doc_df)]

        for _, sample in samples:

            tok_sample_id = np.full(sample.index.shape, fill_value = self._n_samples)
            sample["sample_id"] = tok_sample_id

            if span_labels:
                sample = self._label_spans(sample, span_labels)
            
            if self._need_bio:
                sample = self._label_bios(sample)
            
            self._fuse_subtasks(sample)
            self._encode_labels(sample)
        
            if self.argumentative_markers:
                sample = self._label_ams(sample, mode=self.am_extraction)
                
            seg_length = len(sample.groupby("seg_id", sort = False))
            if self.prediction_level == "seg" and seg_length == 0:
                continue
            
            pretrained_features = self._extract_pretrained_features(sample)

            if "seg_embs" in pretrained_features:
                self._psf_storage.append(pretrained_features["seg_embs"])

            if "word_embs" in pretrained_features:
                self._pwf_storage.append(pretrained_features["word_embs"])

            sample.index = tok_sample_id
            self._df_storage.append(f"df_{self._n_samples}", sample)
            self._n_samples += 1
        






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

