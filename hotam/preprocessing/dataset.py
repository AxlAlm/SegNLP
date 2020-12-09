#basic
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from IPython.display import display
from typing import Union, List, Dict, Tuple
from random import random
import copy
import json
from pathlib import Path
import time
import glob
import hashlib
import shutil
import shelve
from copy import deepcopy

#hotam
import hotam
from hotam.utils import ensure_numpy, load_pickle_data, pickle_data, to_tensor, one_tqdm
from hotam import get_logger
from hotam.preprocessing.encoder import DatasetEncoder
from hotam.preprocessing.preprocessor import Preprocessor
from hotam.preprocessing.labler import Labler
from hotam.preprocessing.split_utils import SplitUtils

#pytorch lightning
import pytorch_lightning as ptl

#pytroch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler
import torch

logger = get_logger("DATASET")

import warnings


class Batch(dict):

    def __init__(self, input_dicts:dict, k2dtype:dict, max_seq:int, tasks:list):
        super().__init__()
        self.tasks = tasks
    
        for k in k2dtype.keys():
            try:
                h = np.stack([s[k] for s in input_dicts])
                
                if len(h.shape) > 1:
                    h = h[:, :max_seq]
                
            except ValueError as e:
                h = [s[k] for s in input_dicts]

            #print(k,h)
            data = to_tensor(h, dtype=k2dtype[k])
            self[k] = data


    def to(self, device):
        self.device = device
        for k,v in self.items():
            if torch.is_tensor(v):
                self[k] = v.to(self.device)

        return self
    

    def change_pad_value(self, new_value):
        for k,v in self.items():
            if k in self.tasks:
                self[k] = v[~v] = new_value


class DataSet(ptl.LightningDataModule, DatasetEncoder, Preprocessor, Labler, SplitUtils):    


    def __init__(self, name, data_path:str="", device:"cpu"=None):
        super().__init__()
        self._name = name
        self.device = device

        if os.path.exists(data_path):
            self.load(data_path)

        self.level2parents = {
                                "token": ["sentence", "paragraph", "document"],
                                "sentence": ["paragraph", "document"],
                                "paragraph": ["document"],
                                "document": []
                                }

        self.parent2children = {
                                "document": ["paragraph","sentence", "token"],
                                "paragraph": ["sentence", "token"],
                                "sentence": ["token"],
                                "token": []
                                }
        self.encoders = {}


    def __getitem__(self, key):
        

        if hotam.preprocessing.settings["CACHE_SAMPLES"]:
            samples_data = self.__load_samples(key)
        
            if None in samples_data:
                samples_data = self.__extract_sample_data(key)
                self.__store_samples(key, samples_data)
        else:
            samples_data = self.__extract_sample_data(key)

        samples_data = sorted(samples_data, key=lambda x:x["lengths"], reverse=True)

        k2dtype = {}
        for k in samples_data[0].keys():

            if k in self.features + ["word_embs"]:
                k2dtype[k] = torch.float
            elif "mask" in k:
                k2dtype[k] = torch.uint8
            else:
                k2dtype[k] = torch.long
        
        # return samples_data
        return Batch(deepcopy(samples_data), k2dtype, max_seq=samples_data[0]["lengths"], tasks=self.tasks)
    

    def __len__(self):
        return self.nr_samples


    @property
    def name(self):
        return self._name

    
    def __extract_sample_data(self, key):

        samples = self.level_dfs["token"].loc[key,].groupby(self.sample_level+"_id")
        #samples = sorted(samples, key=lambda x:len(x), reverse=True)

        sample_data = []
        for i, sample in samples:
            sample_dict = {"ids": i, "lengths":len(sample) if self.prediction_level != "ac" else len(sample.groupby("ac_id"))}
            sample_dict.update(self.__get_text(sample))
            sample_dict.update(self.__get_labels(sample))
            sample_dict.update(self.__get_encs(sample))
            sample_dict.update(self.__get_feature_data(sample))

            if self.prediction_level == "ac":
                sample_dict.update(self.__get_am_ac_spans(sample))

            sample_data.append(sample_dict)

        return sample_data
  

    def __load_samples(self, key):
        d = self.__TMP_SAMPLE_MEMMAP[key]
        return d
        
    
    def __store_samples(self, key, samples_data):
        self.__TMP_SAMPLE_MEMMAP[key] = samples_data


    @one_tqdm(desc="setting up memmaping for cache samples")
    def __setup_cache(self):
        feature_str = "".join(self.features)
        enc_str = "".join(self.encodings)
        tasks_str = "".join(self.all_tasks)
        big_str = feature_str + tasks_str + enc_str + self.sample_level + self.prediction_level + self._name
        hash_obj = hashlib.md5(big_str.encode())
        h = hash_obj.hexdigest()

        base_dir = "/tmp/"
        dirpath = f"{base_dir}{h}-MEMMAP-TMP-hotam/"

        # we remove temp memmap for previous experiments so we dont take up too much space
        dir_exists = os.path.exists(dirpath)
        if not dir_exists:
            dirs = glob.glob(f"{base_dir}*MEMMAP-TMP-hotam")
            for d in dirs:
                shutil.rmtree(d)
            
            os.makedirs(dirpath)

        memmap_file = f"{dirpath}sample_memmap.dat"

        self.__TMP_SAMPLE_MEMMAP = np.memmap(
                                            memmap_file, 
                                            dtype="O", 
                                            mode="w+", 
                                            shape=self.nr_samples
                                            )


    def __get_text(self, sample):
        sample_text = {"text": ""}

        if self.prediction_level == "ac":
            acs = sample.groupby(f"ac_id")
            text = []
            for _, ac in acs:
                text.append(ac["text"].to_numpy())

            sample_text["text"] = text
        else:
            sample_text["text"] = sample["text"].to_numpy()
        
        return sample_text


    def __get_shape(self, ac:bool=None, token:bool=None, char:bool=None, feature_dim:int=None):

        shape = []

        if ac:
            shape.append(self.max_nr_seq)

        if token:
            shape.append(self.max_nr_toks)
        
        if char:
            shape.append(self.encoders["chars"].max_word_length)

        if feature_dim:
            shape.append(feature_dim)
        
        return tuple(shape)


    def __get_labels(self, sample):

        sample_labels = {}
        for task in self.all_tasks:
            task_matrix = np.zeros(self.max_nr_seq)

            if self.prediction_level == "ac":
                ac_i = 0
                acs = sample.groupby(f"ac_id")
                for _, ac in acs:
                    task_matrix[ac_i] = np.nanmax(ac[task].to_numpy())
                    ac_i += 1
            else:
                task_matrix[:sample.shape[0]]  = sample[task].to_numpy()

            sample_labels[task] = task_matrix
        
        return sample_labels
    
    
    def __get_encs(self, sample):

        sample_encs = {}
        for enc in self.encodings:

            shape = self.__get_shape(
                                    ac=self.prediction_level == "ac" and self.tokens_per_sample,
                                    token=enc in set(["words", "bert_encs", "chars"]),
                                    char=enc == "chars",
                                    feature_dim=None
                                    )
            sample_m = np.zeros(shape)

            if self.prediction_level == "ac":
                acs = sample.groupby(f"ac_id")
                for ac_i,(_, ac ) in enumerate(acs):
                    sample_m[ac_i][:ac.shape[0]] = np.stack(ac[enc].to_numpy())

            else:
                sample_m[:sample.shape[0]] = np.stack(sample[enc].to_numpy())
            
            sample_encs[enc] = sample_m
        
        return sample_encs


    def __get_feature_data(self, sample):
        

        feature_dict = {}
        word_embeddings = []
        doc_embeddings = []
        masks_not_added = True

        if self.prediction_level == "ac" and self.tokens_per_sample:
            feature_dict["am_mask"] = np.zeros((self.max_nr_seq, self.max_nr_toks))
            feature_dict["ac_mask"] = np.zeros((self.max_nr_toks, self.max_nr_toks))
        else:
            feature_dict["mask"] = np.zeros(self.max_nr_toks)

        masks_added = False
        
        for feature, fm in self.feature2model.items():
            
            if hotam.preprocessing.settings["STORE_FEATURES"]:
                self.__feature_store_setup(fm)
                    
                ## FOR FEATURE SAVING AND RETREIVING
                #if not hasattr(fm, "_save_features"):


            shape = self.__get_shape(
                                    ac=self.prediction_level == "ac" and self.tokens_per_sample,
                                    token=fm.level == "word",
                                    #char=enc == "chars",
                                    feature_dim=fm.feature_dim
                                    )
            feature_matrix = np.zeros(shape)

            sample_length = sample.shape[0]

            if self.prediction_level != "token":

                if self.prediction_level == "ac":
                    
                    feature_matrix, am_mask, ac_mask = self.__extract_ADU_features(fm, sample, sample_shape=feature_matrix.shape)
                    
                    if self.__word_features:
                        if feature in self.__word_features and not masks_added:
                            feature_dict["am_mask"] = am_mask
                            feature_dict["ac_mask"] = ac_mask 
                            feature_dict["mask"] = np.max(ac_mask, axis=-1)
                            masks_added = True
            else:

                if fm.context and fm.context != self.sample_level:

                    start1 = time.time()
                    contexts = sample.groupby(f"{fm.context}_id")

                    sample_embs = []
                    for _, context_data in contexts:
                        sample_embs.extend(fm.extract(context_data)[:context_data.shape[0]])

                    if fm.level == "word":
                        feature_matrix[:sample_length] = np.array(sample_embs)
                    else:
                        raise NotImplementedError
                
                else:
                    feature_matrix[:sample_length] = fm.extract(sample)[:sample_length]
                
                feature_dict["mask"][:sample_length] = np.ones(sample_length)


            if fm.level == "word":
                word_embeddings.append(feature_matrix)
            
            if fm.level == "doc":
                doc_embeddings.append(feature_matrix)


        if len(word_embeddings) > 1:
            feature_dict["word_embs"] = np.concatenate(word_embeddings, axis=-1)
        elif len(word_embeddings) == 1:
            feature_dict["word_embs"] = word_embeddings[0]
    
        if len(doc_embeddings) > 1:
            feature_dict["doc_embs"] = np.concatenate(doc_embeddings, axis=-1)
        elif len(doc_embeddings) == 1:
            feature_dict["doc_embs"] = doc_embeddings[0]
        
        return feature_dict


    def __extract_ADU_features(self, fm, sample, sample_shape=None):

        sentences  = sample.groupby("sentence_id")

        sample_matrix = np.zeros(sample_shape)
        ac_mask_matrix = np.zeros(sample_shape[:-1])
        am_mask_matrix = np.zeros(sample_shape[:-1])


        ac_i = 0
        for _, sent in sentences:

            sent.index = sent.pop("sentence_id")
            
            #for word embeddings we use the sentence as context (even embeddings which doesnt need contex as
            # while it takes more space its simpler to treat them all the same).
            # we get the word embeddings then we index the span we want, e.g. Argumnet Discourse Unit
            # as these are assumed to be within a sentence
            if fm.level == "word":
                sent_word_embs = fm.extract(sent)[:sent.shape[0]]                
                acs = sent.groupby("ac_id")

                for ac_id , ac in acs:

                    am_mask = sent["am_id"].to_numpy() == ac_id
                    ac_mask = sent["ac_id"].to_numpy() == ac_id
                    adu_mask = am_mask + ac_mask

                    am_mask = am_mask[adu_mask]
                    ac_mask = ac_mask[adu_mask]

                    adu_embs = sent_word_embs[adu_mask]
                    adu_len = adu_embs.shape[0]

                    sample_matrix[ac_i][:adu_len] = sent_word_embs[adu_mask]
                    ac_mask_matrix[ac_i][:adu_len] = ac_mask.astype(np.int8)
                    am_mask_matrix[ac_i][:adu_len] = am_mask.astype(np.int8)
                    ac_i += 1



                
            # for features such as bow we want to pass only the Argument Discourse Unit
            else:
                acs = sent.groupby("ac_id")
                for ac_id, ac in acs:
                    sent.index = sent["id"]
                    am = sent[sent["am_id"] == ac_id]
                    ac = sent[sent["ac_id"] == ac_id]
                    adu = pd.concat((am,ac))
                    adu.index = adu.pop("ac_id")
                    sample_matrix[ac_i] = fm.extract(adu)
                    ac_i += 1

        return sample_matrix, am_mask_matrix, ac_mask_matrix


    def __get_am_ac_spans(self, sample):
        """
        for each sample we get the spans of am, ac and the whole adu.
        if there is no am, we still add an am span to keep the am and ac spans
        aligned. But we set the values to 0 and start the adu from the start of the ac instead
        of the am.

        NOTE!! that the end

        """
        
        am_spans = []
        ac_spans = []
        adu_spans = []

        ac_goups = sample.groupby("ac_id")

        for ac_id, gdf in ac_goups:
            
            am = sample[sample["am_id"]==ac_id]
            
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

                #print(am, min(am[f"{self.sample_level}_token_id"]), max(am[f"{self.sample_level}_token_id"]))

            ac_start = min(gdf[f"{self.sample_level}_token_id"])
            ac_end = max(gdf[f"{self.sample_level}_token_id"])
            ac_span = (ac_start, ac_end)

            if has_am:
                adu_span = (am_start, ac_end)
            else:
                adu_span = (ac_start, ac_end)

            am_spans.append(am_span)
            ac_spans.append(ac_span)
            adu_spans.append(adu_span)

        return {"am_spans":am_spans, "ac_spans":ac_spans, "adu_spans":adu_spans}


    def __feature_store_setup(self, fm):
            
        if fm.level == "word":
            nr_sentence = len(self.level_dfs["token"]["sentence_id"].unique())
            shape = (nr_sentence, self.__get_nr_tokens("sentence"), fm.feature_dim)
        
        elif self.prediction_level == "ac":
            body, suffix = self._feature2memmap[feature].rsplit(".",1)
            self._feature2memmap[feature] = body + "_ac_" + suffix

            nr_acs = int(self.level_dfs["token"]["ac_id"].max())
            shape = (nr_acs, fm.feature_dim)

        else:
            raise NotImplementedError("Not supported yet")
            
        fm._init_feature_save(
                                dataset_name = self.name,
                                feature_name = fm.name,
                                #memmap_file=self._feature2memmap[feature], 
                                shape = shape,
                                dtype = fm.dtype,
                                )

    def stats(self):
    
        column_data = {"type":[], "train": [], "val":[], "test": []}

        for i, split_set in self.splits.items():

            for s, ids in split_set.items():
                
                if self.nr_splits == 1:
                    name = self.sample_level
                else:
                    name = f"{self.sample_level}_{i}"

                # row = {"type": name, s:len(ids)}

                if name not in column_data["type"]:
                    column_data["type"].append(name)

                column_data[s].append(len(ids))


                for task in self.tasks:
                    df = self.level_dfs["token"]
                    label_counts = dict(df[df[f"{self.sample_level}_id"].isin(ids)][task].value_counts())

                    if self.prediction_level == "token":
                        label_counts = {self.decode(l,task):c for l,c in label_counts.items()}
                    else:
                        label_counts = {self.decode(l,task):int(c/len(ids)) for l,c in label_counts.items()}

                    label_counts.update({l:0 for l in self.task2labels[task] if l not in label_counts.keys()})
                    label_counts = dict(sorted(list(label_counts.items()), key=lambda x:x[0]))

                    for l,nr in label_counts.items():
                        
                        if self.nr_splits == 1:
                            l_name = l
                            #l_name = f"{task}_{l}"
                        else:
                            l_name = f"{l}_{i}"
                            #l_name = f"{task}_{l}_{i}"

                        if l_name not in column_data["type"]:
                            column_data["type"].append(l_name)

                        column_data[s].append(nr)
            
        


        df = pd.DataFrame(column_data)
        df.index = df.pop("type")
        return df


    def show(self, jupyter:bool=False):
        """
        prints the dataframes for each level

        Parameters
        ----------
        jupyter : bool, optional
            if you are working in jupyter set True for nicer printing, by default False

        Raises
        ------
        RuntimeError
            if there is no data
        """

        if not hasattr(self, "level_dfs"):
            raise RuntimeError("Dataset contain no data. Use add_sample() or add_samples() to populate dataset")

        for level, df in self.level_dfs.items():
            #df.reset_index(drop=True, inplace=True)
            if jupyter:
                display(df.head(5))
            else:
                logger.info(f'DataFrame of level {level}- \n{df.head(5)}')


        if hasattr(self, "exp_df"):
            if jupyter:
                display(self.exp_df.head(5))
            else:
                logger.info(f'DataFrame of EXP DF - \n{self.exp_df.head(5)}')


    def dataset_as_tokens(self):
        #return self.level_dfs["token"].T.to_dict()
       return [dict({"dataset":self.name},**token_dict) for i,token_dict in self.level_dfs["token"].iterrows()]


    @one_tqdm(desc="Saving Preprocessed Dataset")
    def save(self, pkl_path:str):
        """saves the preprocessed dataset, save dataset_level

        Parameters
        ----------
        pkl_path : str
            string path
        """
        pickle_data([self.level_dfs, self.dataset_level], pkl_path)


    @one_tqdm(desc="Loading Preprocessed Dataset")
    def load(self, pkl_path:str):
        """
        loads the data and creates the stacks for preprocessing (if one wanted to add data)

        Parameters
        ----------
        pkl_path : 
            path string

        Raises
        ------
        RuntimeError
            if data already exist it will break loading.
        """
        if hasattr(self, "level_dfs"):
            raise RuntimeError("data already exist. Overwriting is currently unsupported with load()")

        self.level_dfs, self.dataset_level = load_pickle_data(pkl_path)
        self.stack_level_data = {l:[] for l in self.level_dfs.keys()}


    @one_tqdm(desc="Save Encoded State")
    def __save_enc_state(self):

        state_dict = {}
        state_dict["data"] = self.level_dfs["token"].to_dict()
        state_dict["all_tasks"] = self.all_tasks
        state_dict["subtasks"] = self.subtasks
        state_dict["main_tasks"] = self.main_tasks
        state_dict["task2subtasks"] = self.task2subtasks
        state_dict["task2labels"] = self.task2labels

        with open(self._enc_file_name,"w") as f:
            json.dump(state_dict, f)


    @one_tqdm(desc="Load Encoded State")
    def __load_enc_state(self):
        with open(self._enc_file_name,"r") as f:
            state_dict = json.load(f)

        self.level_dfs["token"] = pd.DataFrame(state_dict["data"])
        self.all_tasks = state_dict["all_tasks"]
        self.subtasks = state_dict["subtasks"]
        self.task2subtasks = state_dict["task2subtasks"]
        self.task2labels = state_dict["task2labels"]
        self.main_tasks = state_dict["main_tasks"]


    @one_tqdm(desc="Finding task and subtasks")
    def __fix_tasks(self):
        
        subtasks_set = set()
        self.task2subtasks = {}

        #labels are collected subtask and tasks
        all_task = set()

        all_task.update(self.tasks)

        for task in self.tasks:
            
            subtasks = task.split("_")

            if len(subtasks) <= 1:
                break
        
            self.task2subtasks[task] = subtasks

            subtasks_set.update(subtasks)
            all_task.update(subtasks)

        
        self.all_tasks = sorted(list(all_task))
        self.subtasks = sorted(list(subtasks))


    @one_tqdm(desc="Getting task labels")
    def __get_task2labels(self):
        self.task2labels = {}
        for task in self.all_tasks:
            self.task2labels[task] = list(self.level_dfs["token"][task].unique())

        # if we only predict on acs we dont need "None" label
        if self.prediction_level == "ac":
            self.task2labels["ac"].remove("None")
        
    
    @one_tqdm(desc="Reformating Labels")
    def __fuse_subtasks(self):

        for task in self.tasks:
            subtasks = task.split("_")

            if len(subtasks) <= 1:
                continue

            subtask_labels  = self.level_dfs["token"][subtasks].apply(lambda row: '_'.join([str(x) for x in row]), axis=1)
            self.level_dfs["token"][task] = subtask_labels


    def __get_nr_tokens(self, level):
        """
        Gets the max number of words for each of the levels in the heirarchy. E.g. max words in documents, 
        sentences, paragraphs.

        """
        return max(len(g) for i,g in self.level_dfs["token"].groupby(level+"_id"))
  

    def __get_max_nr_seq(self):
        samples = self.level_dfs["token"].groupby(self.sample_level+"_id")
        return max(len(sample.groupby(self.prediction_level+"_id")) for i, sample in samples)


    @one_tqdm(desc="Removing Duplicates")
    def remove_duplicates(self):
        """
        Removes duplicate on the dataset level based on the text and also updates the splits to match the pruned dataset.

        text filed which is used to check duplicates is the original text.

        """


        def __get_duplicate_set(df):

            # TODO: solve this with groupby isntead???
            duplicate_mask_all = df.duplicated(subset="text",  keep=False).to_frame(name="bool")
            duplicate_ids_all = duplicate_mask_all[duplicate_mask_all["bool"]==True].index.values

            dup_ids = df.loc[duplicate_ids_all]["id"]
            dup_items = df.loc[duplicate_ids_all]["text"]
            dup_dict = dict(enumerate(set(dup_items)))
            dup_pairs = {i:[] for i in range(len(dup_dict))}
            for i, item in dup_dict.items():
                for ID, item_q in zip(dup_ids, dup_items):
                    if item == item_q:
                        dup_pairs[i].append(ID)

            return list(dup_pairs.values())

        target_level = self.dataset_level
        df = self.level_dfs[target_level]
        duplicate_sets = __get_duplicate_set(df)
        duplicate_ids = np.array([i for duplicate_ids in duplicate_sets for i in  duplicate_ids[1:]])

        df = df[~df["id"].isin(duplicate_ids)]
        df.reset_index(drop=True, inplace=True)
        self.level_dfs[target_level] = df


        for level, df in self.level_dfs.items():

            if level == target_level:
                continue

            df = df[~df[f"{target_level}_id"].isin(duplicate_ids)]
            df.reset_index(drop=True, inplace=True)
            self.level_dfs[level] = df

        
        self.duplicate_ids = duplicate_ids

        if hasattr(self, "splits"):
            self.update_splits()

        logger.info(f"Removed {len(duplicate_ids)} duplicates from dataset. Duplicates: {duplicate_sets}")


    def get_subtask_position(self, task:str, subtask:str) -> int:
        """fetches the position of the subtask in the the task label. Return the index of the subtask in the task.

        Parameters
        ----------
        task : str
            task
        subtask : str
            subtask

        Returns
        -------
        int
            id for the subtask in the task
            
        """
        return self.task2subtasks[task].index(subtask)


    def setup(  self,
                tasks:list,
                multitasks:list, 
                prediction_level:str,
                sample_level:str, 
                features:list,
                encodings:list,
                remove_duplicates:bool=True,
                tokens_per_sample:bool=False
                ):
        """prepares the data for an experiemnt on a set task.

        adds additional labels to task
        set encodings and sample_level

        Parameters
        ----------
        sample_level : str
            level on which the samples should be set
        multitasks : list
            if there are more labels 
        encodings : list
            encodings to be made, e.g. token, pos, bert
        """
        self.prediction_level = prediction_level
        self.sample_level = sample_level
        self.tokens_per_sample = tokens_per_sample

        self.main_tasks = tasks
        self.tasks = tasks + multitasks
        self.encodings = encodings

        self.feature2model = {fm.name:fm for fm in features}
        self.features = list(self.feature2model.keys())
        self.feature2dim = {fm.name:fm.feature_dim for fm in features}
        self.feature2dim["word_embs"] = sum(fm.feature_dim for fm in features if fm.level == "word")
        self.feature2dim["doc_embs"] = sum(fm.feature_dim for fm in features if fm.level == "word")
        self.__word_features = [fm.name for fm in self.feature2model.values() if fm.level == "word"]

        #create a hash encoding for the exp config
        #self._exp_hash = self.__create_exp_hash()
        self._enc_file_name = os.path.join("/tmp/", f"{'_'.join(self.tasks+self.encodings)+self.prediction_level}_enc.json")

        if remove_duplicates:
            self.remove_duplicates()

        #remove duplicates from splits
        if remove_duplicates:
            if self.duplicate_ids.any():
                self.update_splits()
        
        enc_data_exit = os.path.exists(self._enc_file_name)
        if enc_data_exit:
            self.__load_enc_state()
            self._create_data_encoders()
            self._create_label_encoders()
    
        else:
            self.__fix_tasks()
            self._create_data_encoders()
            self._encode_data() 
            self.__fuse_subtasks()
            self.__get_task2labels()
            self._create_label_encoders()
            self._encode_labels()
            self.__save_enc_state()
        
        
        self.level_dfs["token"].index = self.level_dfs["token"][f"{sample_level}_id"].to_numpy() #.pop(f"{sample_level}_id")

        if self.prediction_level == "token":
            self.max_nr_toks = self.__get_nr_tokens(self.sample_level)
            self.max_nr_seq = self.max_nr_toks
        else:
            self.max_nr_toks = self.__get_nr_tokens(self.prediction_level)
            self.max_nr_seq = self.__get_max_nr_seq()
        
        self.nr_samples = self.level_dfs[self.sample_level].shape[0]

        self._change_split_level()

        self.config = {
                        "prediction_level":prediction_level,
                        "sample_level": sample_level,
                        "tasks": tasks,
                        "multitasks": multitasks,
                        "encodings": encodings,
                        "features": self.features,
                        "remove_duplicates": remove_duplicates,
                        "task_labels": self.task2labels
                    }

        if hotam.preprocessing.settings["CACHE_SAMPLES"]:
            self.__setup_cache()


    def train_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["train"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0]) #, shuffle=True)


    def val_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["val"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0]) #, shuffle=True)


    def test_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["test"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0])

