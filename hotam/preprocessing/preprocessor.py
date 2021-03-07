#basic
import numpy as np
from tqdm import tqdm
import os
import webbrowser
import pandas as pd
from IPython.display import display, Image
from typing import Union, List, Dict, Tuple
import random
import copy
import json
from pathlib import Path
import time
import glob
import hashlib
import shutil
import shelve
from copy import deepcopy
from pprint import pprint
import warnings
from time import time
import itertools


#hotam
import hotam
from hotam.utils import ensure_numpy, load_pickle_data, pickle_data, to_tensor, one_tqdm, timer, dynamic_update
from hotam import get_logger

from .encoder import Encoder
from .textprocessor import TextProcesser
from .labeler import Labeler
from .dataset_preprocessor import DataPreprocessor

from hotam.nn import ModelInput

#pytorch lightning
import pytorch_lightning as ptl

#pytroch
import torch


logger = get_logger("DATASET")


class Preprocessor(Encoder, TextProcesser, Labeler, DataPreprocessor):    


    def __init__(self,
                prediction_level:str,
                sample_level:str,
                input_level:str,
                features:list = [],
                encodings:list = [],
                tokens_per_sample:bool=False,
                ):
        super().__init__()

        self.prediction_level = prediction_level
        self.sample_level = sample_level
        self.input_level = input_level
        self.tokens_per_sample = tokens_per_sample

        self.__labeling = False
        self.argumentative_markers = False

        self.encodings = encodings

        self.feature2model = {fm.name:fm for fm in features}
        self.features = list(self.feature2model.keys())
        self._feature_groups = set([fm.group for fm in features])
        self.feature2dim = {fm.name:fm.feature_dim for fm in features}
        self.feature2dim.update({
                                group:sum([fm.feature_dim for fm in features if fm.group == group]) 
                                for group in self._feature_groups
                                })
        #self.__word_features = [fm.name for fm in self.feature2model.values() if fm.level == "word"]


        self._init_preprocessor()
        self._init_encoder()
        self._init_DataPreprocessor()
        self._create_data_encoders()
    
    @property
    def config(self):
        return {
                "tasks": None if not hasattr(self, "tasks") else self.tasks,
                "subtasks": None if not hasattr(self, "subtasks") else self.subtasks,
                "all_tasks": None if not hasattr(self, "all_tasks") else self.all_tasks,
                "task2labels":None if not hasattr(self, "task2labels") else self.task2labels,
                "prediction_level": self.prediction_level,
                "sample_level": self.sample_level,
                "input_level": self.input_level,
                "token_per_sample":self.tokens_per_sample,
                "feature2dim": self.feature2dim,
                "encodings":self.encodings
                }


    def __call__(self, docs:List[Union[dict, str]]) -> ModelInput: #docs:List[str],token_labels:List[List[dict]] = None, span_labels:List[dict] = None):

        Input = ModelInput()
        
        for doc in docs:
                        
            if isinstance(doc,dict):
                span_labels = doc.get("span_labels", None)
                token_labels = doc.get("token_labels", None)
                doc = doc["text"]

            doc_df = self._process_doc(doc)
            doc_id = int(doc_df["id"].to_numpy()[0])

            #everything within this block should be speed up
            if self.__labeling:
                if span_labels:
                    doc_df = self._label_spans(doc_df, span_labels)

                if token_labels:
                    doc_df = self._label_tokens(doc_df, token_labels)
                
                if self.__need_bio:
                    doc_df = self._label_bios(doc_df)
                
                self.__fuse_subtasks(doc_df)
                self._encode_labels(doc_df)


            if self.argumentative_markers:
                doc_df = self._label_ams(doc_df)
            
            if self.encodings:
                self._encode_data(doc_df)


            if self.input_level != self.sample_level:
                samples = doc_df.groupby(f"{self.sample_level}_id")
            else:
                samples = [(doc_id,doc_df)]

            for i, sample in samples:
                Input.add("id", i, None)
                Input.add("lengths", sample.shape[0], "token")
                
                spans_grouped = sample.groupby("span_id")
                Input.add("lengths", len(spans_grouped), "span")
                Input.add("lengths_tok", np.array([g.shape[0] for i, g in spans_grouped]), "span")

                non_span_mask = (~np.isnan(sample.groupby("span_id").first()["unit_id"].to_numpy())).astype(np.uint8)
                Input.add("none_span_mask", non_span_mask, "span")

                if self.prediction_level == "unit":
                    units = sample.groupby("unit_id")
                    Input.add("lengths", len(units), "unit")
                    Input.add("lengths_tok", np.array([g.shape[0] for i, g in units]), "unit")

                # # if hasattr(self, "max_sent"):
                # #     sent_grouped = sample.groupby("sentence_id")
                # #     Input.add("lengths", len(sent_grouped), "sentence")
                # #     Input.add("lengths_tok", [len(g) for i, g in sent_grouped], "sentence")
 
                self.__get_text(Input, sample)
                self.__get_encs(Input, sample)
                self.__get_feature_data(Input, sample)

                if self.__labeling:
                    self.__get_labels(Input, sample)

                if self.prediction_level == "unit":
                    self.__get_am_unit_idxs(Input, sample)

        return Input.to_numpy()
  

    def __get_text(self, Input:ModelInput, sample:pd.DataFrame):

        if self.prediction_level == "unit":
            units = sample.groupby(f"unit_id")
            text = []
            for _, unit in units:
                text.append(unit["text"].to_numpy().tolist())

            Input.add("text", np.array(text, dtype="U30"), "unit")
            #sample_text["text"] = np.array(text)
        else:
            #Input.add("text", sample["text"].to_numpy().astype("S"))
            Input.add("text", sample["text"].to_numpy().astype("U30"), "token")
            #sample_text["text"] = sample["text"].to_numpy()
        
        #return sample_text


    def __get_labels(self, Input:ModelInput, sample:pd.DataFrame):

        sample_labels = {}
        for task in self.all_tasks:

            unit_i = 0
            units = sample.groupby(f"unit_id")
            unit_task_matrix = np.zeros(len(units))
            for unit_id, unit in units:
                unit_task_matrix[unit_i] = np.nanmax(unit[task].to_numpy())
                unit_i += 1

            #if self.prediction_level == "token":
            #task_matrix = np.zeros(len(sample.index))
            #task_matrix[:sample.shape[0]] = sample[task].to_numpy()

            Input.add(task, sample[task].to_numpy().astype(np.int), "token")
            Input.add(task, unit_task_matrix.astype(np.int), "unit")
        
        return sample_labels
    

    def __get_dep_encs(self,  Input:ModelInput, sample:pd.DataFrame):

        deprel_m = np.zeros((self.max_sent, self.max_sent_tok))      
        dephead_m = np.zeros((self.max_sent, self.max_sent_tok))      
        sent2root = np.zeros(self.max_sent)
        sentences  = sample.groupby("sentence_id")

        sentences  = sample.groupby("sentence_id")
        units = sample.groupby("unit_id")
        nr_units  = len(units)
        nr_sents = len(sentences)
        nr_tok_sents = max([len(s) for s in sentences])
        sample_m = np.zeros((nr_sents, nr_tok_sents))      

        unit_i = 0

        #create a mask
        if self.prediction_level == "unit":
            
            #information about which unit belong in which sentence
            unit2sentence = np.zeros(len(units))

            # given information from unit2sentence, we can get the mask for a sentnece for a particular unit
            # so we know what tokens belong to unit and not
            sentence_unit_mask = np.zeros((nr_units, nr_sents))


        for sent_i, (sent_id, sent_df) in enumerate(sentences):

            deprel_m[sent_i][:sent_df.shape[0]] = sent_df["deprel"].to_numpy()
            dephead_m[sent_i][:sent_df.shape[0]] = sent_df["dephead"].to_numpy()

            root_id = self.encode_list(["root"], "deprel")[0]
            root_idx = int(np.where(sent_df["deprel"].to_numpy() == root_id)[0])
            sent2root[sent_i] = root_idx
                            
            #create a mask
            if self.prediction_level == "unit":
                units = sent_df.groupby("unit_id")

                for unit_id, unit_df in units:
                    unit2sentence[unit_i] = sent_i

                    unit_ids = sent_df["unit_id"].to_numpy()
                    mask = unit_ids == unit_id
                    sentence_unit_mask[unit_i][:sent_df.shape[0]] = mask.astype(np.int8)
                    unit_i += 1


        if self.prediction_level == "unit":
            Input.add("unit2sentence",unit2sentence, "sentence")
            Input.add("sent_unit_mask", sentence_unit_mask, "sentence")


        Input.add("sent2root", sent2root, "sentence")
        Input.add("deprel", deprel_m, "sentence")
        Input.add("dephead", dephead_m, "sentence")
  

    def __get_encs(self, Input:ModelInput, sample:pd.DataFrame):

        deps_done = False
        for enc in self.encodings:

            if self.sample_level != "sentence" and enc in ["deprel", "dephead"]:

                if deps_done:
                    continue

                self.__get_dep_encs(Input=Input, sample=sample)

            else:
                if self.prediction_level == "unit" and not self.tokens_per_sample:

                    units = sample.groupby("unit_id")
                    nr_tok_units = max([len(unit) for unit in units])
                    unit_matrix  = np.zeros(len(units), nr_tok_units, dtype=np.int)
                    for unit_i,(_, unit ) in enumerate(units):                        
                        sample_m[unit_i][:unit.shape[0]] = np.stack(unit[enc].to_numpy())

                    Input.add(enc, unit_matrix, "unit")

                else:
                    Input.add(enc, np.stack(sample[enc].to_numpy()).astype(np.int), "token")
            
  
    def __get_feature_data(self, Input:ModelInput, sample:pd.DataFrame):
        
        mask_dict = {}
        feature_dict = {}


        unit_mask_added = False
        token_mask_added = False

        for feature, fm in self.feature2model.items():
    
            if self.prediction_level == "unit":
                if self.tokens_per_sample:
                    if fm.level == "doc":
                        alt1 = True
                    else:
                        alt1 = False
                else:
                    alt1 = True
            else:
                alt1 = False

    
            sample_length = sample.shape[0]

            if alt1:
            #if self.prediction_level == "unit" and not self.tokens_per_sample:
                
                nr_units = len(sample.groupby("unit_id"))

                if fm.level == "word":
                    nr_tok_units = max([len(unit) for unit in units])
                    feature_matrix = np.zeros(nr_units, nr_tok_units, fm.feature_dim)
                else: 
                    fm.feature_dim = np.zeros(nr_units, fm.feature_dim)


                feature_matrix, am_mask, unit_mask = self.__extract_ADU_features(fm, sample, sample_shape=feature_matrix.shape)

                if fm.level == "word" and not unit_mask_token_added:
                    Input.add("token_mask", am_mask.astype(np.uint8), "am")
                    Input.add("token_mask", unit_mask.astype(np.uint8), "unit")
                
                if not unit_mask_token_added:
                    
                    if fm.level == "word":
                        Input.add("mask", np.max(unit_mask, axis=-1).astype(np.uint8), "unit")
                    else:
                        Input.add("mask",  unit_mask.astype(np.uint8), "unit")
           
            else:
                

                if fm.level == "word":
                    # context is for embeddings such as Bert and Flair where the word embeddings are dependent on the surrounding words
                    # so for these types we need to extract the embeddings per context. E.g. if we have a document and want Flair embeddings
                    # we first divide the document up in sentences, extract the embeddigns and the put them bunitk into the 
                    # ducument shape.
                    # Have chosen to not extract flair embeddings with context larger than "sentence".
                    if fm.context and fm.context != self.sample_level:

                        contexts = sample.groupby(f"{fm.context}_id")

                        sample_embs = []
                        for _, context_data in contexts:
                            sample_embs.extend(fm.extract(context_data)[:context_data.shape[0]])

                        # if fm.level == "word":
                            #Input.add(feature_name, np.array(sample_embs))
                            #feature_matrix[:sample_length] = np.array(sample_embs)
                        # else:
                        #     raise NotImplementedError

                        feature_matrix = np.array(sample_embs)
                
                    else:
                        #feature_matrix[:sample_length] = fm.extract(sample)[:sample_length]
                        feature_matrix = fm.extract(sample)[:sample_length]


                    if not token_mask_added:
                        #mask_dict["token_mask"][:sample_length] = np.ones(sample_length)
                        Input.add("mask", np.ones(sample_length, dtype=np.uint8), "token")

                else:
                    feature_matrix = fm.extract(sample)[:sample_length]


            if fm.group not in feature_dict:
                feature_dict[fm.group] = {
                                        "level": "unit" if fm.level == "doc" else "token",
                                        "data":[]
                                        }
            
            feature_dict[fm.group]["data"].append(feature_matrix)


        for group_name, group_dict in feature_dict.items():
            if len(group_dict["data"]) > 1:
                Input.add(group_name, np.concatenate(group_dict["data"], axis=-1), group_dict["level"])
            else:
                Input.add(group_name, group_dict["data"][0], group_dict["level"])


    def __extract_ADU_features(self, fm, sample, sample_shape=None):

        sentences  = sample.groupby("sentence_id")

        sample_matrix = np.zeros(sample_shape)
        unit_mask_matrix = np.zeros(sample_shape[:-1])
        am_mask_matrix = np.zeros(sample_shape[:-1])


        unit_i = 0
        for _, sent in sentences:

            sent.index = sent.pop("sentence_id")
            
            #for word embeddings we use the sentence as context (even embeddings which doesnt need contex as
            # while it takes more spunite its simpler to treat them all the same).
            # we get the word embeddings then we index the unit we want, e.g. Argumnet Discourse Unit
            # as these are assumed to be within a sentence
            if fm.level == "word":
                sent_word_embs = fm.extract(sent)[:sent.shape[0]]  
                units = sent.groupby("unit_id")

                for unit_id , unit in units:

                    am_mask = sent["am_id"].to_numpy() == unit_id
                    unit_mask = sent["unit_id"].to_numpy() == unit_id
                    adu_mask = am_mask + unit_mask

                    am_mask = am_mask[adu_mask]
                    unit_mask = unit_mask[adu_mask]

                    adu_embs = sent_word_embs[adu_mask]
                    adu_len = adu_embs.shape[0]

                    sample_matrix[unit_i][:adu_len] = sent_word_embs[adu_mask]
                    unit_mask_matrix[unit_i][:adu_len] = unit_mask.astype(np.int8)
                    am_mask_matrix[unit_i][:adu_len] = am_mask.astype(np.int8)
                    unit_i += 1
                
            # for features such as bow we want to pass only the Argument Discourse Unit
            else:

                units = sent.groupby("unit_id")
                for unit_id, unit in units:
                    sent.index = sent["id"]
                    am = sent[sent["am_id"] == unit_id]
                    unit = sent[sent["unit_id"] == unit_id]
                    adu = pd.concat((am,unit))
                    adu.index = adu.pop("unit_id")
                    sample_matrix[unit_i] = fm.extract(adu)
                    unit_mask_matrix[unit_i] = 1
                    unit_i += 1

        return sample_matrix, am_mask_matrix, unit_mask_matrix


    def __get_am_unit_idxs(self, Input:ModelInput, sample:pd.DataFrame):
        """
        for each sample we get the units of am, unit and the whole adu.
        if there is no am, we still add an am unit to keep the am and unit units
        aligned. But we set the values to 0 and start the adu from the start of the unit instead
        of the am.

        """
        
        am_units = []
        unit_units = []
        adu_units = []

        unit_goups = sample.groupby("unit_id")

        for unit_id, gdf in unit_goups:
            
            am = sample[sample["am_id"]==unit_id]
            
            has_am = True
            if am.shape[0] == 0:
                am_start = 0
                am_end = 0
                am_span = (am_start, am_end)
                has_am = False
            else:
                am_start = min(am[f"{self.sample_level}_token_id"])
                am_end = max(am[f"{self.sample_level}_token_id"])
                am_span = (am_start, am_end)

            unit_start = min(gdf[f"{self.sample_level}_token_id"])
            unit_end = max(gdf[f"{self.sample_level}_token_id"])
            unit_span = (unit_start, unit_end)

            if has_am:
                adu_span = (am_start, unit_end)
            else:
                adu_span = (unit_start, unit_end)

            am_units.append(am_span)
            unit_units.append(unit_span)
            adu_spans.append(adu_span)

        Input.add("unit_idx", np.array(am_spans), "am")
        Input.add("unit_idx", np.array(unit_span), "span")
        Input.add("unit_idx", np.array(adu_spans), "adu")

        #return {"am_spans":am_spans, "span_spans":span_spans, "adu_spans":adu_spans}


    def __fuse_subtasks(self, df):

        for task in self.tasks:
            subtasks = task.split("+")
            
            if len(subtasks) <= 1:
                continue

            subtask_labels  = df[subtasks].apply(lambda row: '_'.join([str(x) for x in row]), axis=1)
            df[task] = subtask_labels


    def __get_subtasks(self, tasks):
        subtasks = []
        for task in tasks:
            subtasks.extend(task.split("+"))
        return subtasks


    def __get_task_labels(self, task, task_labels):
        
        task2labels = {}
        for task in task:

            subtasks = task.split("+")

            label_groups = []
            has_seg = False
            for st in subtasks:
                task2labels[st] = task_labels[st]

                if st == "seg":
                    BIO = task2labels["seg"].copy()
                    BIO.remove("O")
                    label_groups.append(BIO)
                    has_seg = True
                else:
                    label_groups.append(task2labels[st])

            combs = list(itertools.product(*label_groups))

            if has_seg:
                none_label = ["O"] + ["None" if s != "link" else "0" for s in task2labels if s != "seg"]
                combs.insert(0,none_label)

            task2labels[task] = ["_".join([str(c) for c in comb]) for comb in combs]
        
        return task2labels


    def expect_labels(self, tasks:list, task_labels:dict):
        self.__need_bio = False
        self.__labeling = True
        self.tasks = tasks
        self.subtasks = self.__get_subtasks(tasks)
        self.all_tasks = sorted(set(tasks + self.subtasks))
        self.__need_bio = "seg" in self.subtasks

        if self.__need_bio:
            task_labels["seg"] = ["O","B","I"]
        
        self.task2labels = self.__get_task_labels(tasks, task_labels)
        self._create_label_encoders()


