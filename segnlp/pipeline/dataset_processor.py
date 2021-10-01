


#basics
from time import sleep
from tqdm.auto import tqdm
import numpy as np
from typing import Union, Sequence
import os
from collections import Counter
import pandas as pd
import shutil

# segnlp
from segnlp.datasets.base import DataSet
import segnlp.utils as utils
from segnlp import get_logger

logger = get_logger("DatasetPreprocessor")



class DatasetProcessor:


    def _init_dataset_processor(self):

        # argumentative markers
        self.argumentative_markers : bool = False 
        if "am" in self.other_levels:
            self.argumentative_markers = True

            if self.dataset_name == "MTC":
                self.am_extraction = "from_list"
            else:
                self.am_extraction = "pre"

        # preprocessing
        self._need_bio : bool = "seg" in self.subtasks
        self._labeling : bool = True
        self._removed : int = 0


    def _process_dataset(self, dataset : DataSet) -> pd.DataFrame:

        self._n_samples = 0

        sample_dfs = []

        for i in tqdm(range(len(dataset)), desc=f"Prerocessing Dataset (nlp = {self._nlp_name})"):
            doc = dataset[i]

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
                sample["sample_token_id"] = sample[f"{self.sample_level}_token_id"].to_numpy()

                if span_labels:
                    sample = self._label_spans(sample, span_labels)
                
                if self._need_bio:
                    sample = self._label_bios(sample)
                
                sample = self._fuse_subtasks(sample)
                sample = self._encode_labels(sample)

                if self.argumentative_markers:
                    sample = self._label_ams(sample, mode=self.am_extraction)
                

                if self.sample_level != "sentence":

                    # remove samples that doens have segments if we are predicting on segments
                    segs = sample.groupby("seg_id", sort = False)
                    seg_length = len(segs)
                    if self.prediction_level == "seg" and seg_length == 0:
                        continue

                    # it might be helpfult to keep track on the global seg_id of the target
                    # i.e. the seg_id of the linked segment
                    seg_first = segs.first()
                    target_ids = seg_first.index.to_numpy()[seg_first["link"].to_numpy()]
                    sample.loc[~sample["seg_id"].isna() ,"target_id"] = np.repeat(target_ids, segs.size().to_numpy())
                

                sample.index = tok_sample_id
                sample_dfs.append(sample)
                self._n_samples += 1

        #concatenate all samples to a big dataframe
        df = pd.concat(sample_dfs)

        #save csv 
        df.to_csv(self._path_to_df)

        return df
