#basic
from typing import Union, List, Dict, Tuple
import numpy as np
import os
import pandas as pd
from time import time

#segnlp
from segnlp import get_logger
from segnlp.utils import Input


logger = get_logger("DocProcessor")


class DocProcessor:    


    def _process_doc(self, doc:dict): #docs:List[str],token_labels:List[List[dict]] = None, span_labels:List[dict] = None):
        #input = Input()

        span_labels = doc.get("span_labels", None)
        doc = doc["text"]   

        doc_df = self._process_text(doc)

        if self.input_level != self.sample_level:
            samples = doc_df.groupby(f"{self.sample_level}_id", sort=False)
        else:
            samples = [(None, doc_df)]

        for _, sample in samples:

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
   

            pretrained_features = self.__get_pretrained_features()

            if "seg_embs" in pretrained_features:
                self._psf_storage.append(pretrained_features["seg_embs"])

            if "word_embs" in pretrained_features:
                self._pwf_storage.append(pretrained_features["word_embs"])

            sample.index = [i]*len(sample.index)
            store.append("df", sample)

            self._storage.append("df", sample)

        

            # print(sample)

            # file = '/tmp/TEST_H5HPY_DATA.hdf5'
            # # store = pd.HDFStore(file)

            # # for i in range(2000):
            # #     sample.index = [i]*len(sample.index)
            #     store.append("df", sample)

            # print("APPENDING DONE")
            
            # start = time()
            # batch = store.select("df", "index in ['0','400','350']")
            # end = time()
            # print(end - start)
            # print(batch)

    # def _inference_processing(self, doc):

    #     doc_df = self._process_text(doc)

    #     if self.argumentative_markers:
    #         sample = self._label_ams(sample, mode=self.am_extraction)
        
    #     pretrained_features = self.__get_pretrained_features()




    def deactivate_labeling(self):
        self._labeling = False


    def activate_labeling(self):
        self._labeling = True









