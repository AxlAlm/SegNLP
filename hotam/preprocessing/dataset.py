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


#hotam
import hotam
from hotam.utils import ensure_numpy, load_pickle_data, pickle_data, to_tensor, one_tqdm, timer
from hotam import get_logger
from .encoder import DatasetEncoder
from .preprocessor import Preprocessor
from .labeler import Labeler
from .split_utils import SplitUtils

#pytorch lightning
import pytorch_lightning as ptl

#pytroch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler
import torch

#hotviz
from hotviz import hot_tree, hot_text

logger = get_logger("DATASET")


class Batch(dict):


    def __init__(self, input_dict, tasks:list, prediction_level:str, length=int):
        super().__init__(input_dict)
        self.tasks = tasks
        self.prediction_level = prediction_level
        self._len = length
        self.current_epoch = None

    def __len__(self):
        return self._len


    def to(self, device):
        self.device = device
        for k,v in self.items():
            if torch.is_tensor(v):
                self[k] = v.to(self.device)

        return self
    

    def change_pad_value(self, new_value):
        for task in self.tasks:
            self[task][~self[f"{self.prediction_level}_mask"].type(torch.bool)] = -1
     

class DataSet(ptl.LightningDataModule, DatasetEncoder, Preprocessor, Labeler, SplitUtils):    


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
        self.__cutmap = {"doc_embs":"seq", "sent2root":"sent", "ac2sentence":"sent", "sent_ac_mask":"sent"}
        self._setup_done = False


    def __getitem__(self, key):

        if isinstance(key, int):
            key = [key]
        

        if hotam.preprocessing.settings["CACHE_SAMPLES"]:
            samples_data = self.__load_samples(key)
        
            if None in samples_data:
                samples_data = self.__extract_sample_data(key)
                self.__store_samples(key, samples_data)
        else:
            samples_data = self.__extract_sample_data(key)


        if self.prediction_level == "token":
            samples_data = sorted(samples_data, key=lambda x:x["lengths_tok"], reverse=True)
            max_tok = samples_data[0]["lengths_tok"]
        else:
            samples_data = sorted(samples_data, key=lambda x:x["lengths_seq"], reverse=True)
            max_seq = samples_data[0]["lengths_seq"]
            max_tok = max(s["lengths_tok"] for s in samples_data)
            max_seq_tok = max([max(s["lengths_seq_tok"]) for s in samples_data])

        if hasattr(self, "max_sent"):
            max_sent = max(s["lengths_sent"] for s in samples_data)
            max_sent_tok = max([max(s["lengths_sent_tok"]) for s in samples_data])
        
        #unpack and turn to tensors
        unpacked_data = {}
        k2dtype = { 
                    "word_embs":torch.float, 
                    "doc_embs": torch.float, 
                    "ac_mask":torch.uint8,
                    "token_mask":torch.uint8,
                    "am_token_mask":torch.uint8,
                    "am_token_mask":torch.uint8,
                    }

        for k in samples_data[0].keys():
            try:
                h = np.stack([s[k] for s in samples_data])
                cut = self.__cutmap.get(k, "seq" if self.prediction_level == "ac" else "tok")
    
                if cut == "seq":

                    if len(h.shape) == 2:
                        h = h[:, :max_seq]

                    elif len(h.shape) > 2:
                        if "embs" in k:
                            h = h[:, :max_seq, :]
                        else:                        
                            h = h[:, :max_seq, :max_seq_tok]

                elif cut == "tok":

                    #print(k, h , h.shape, max_tok)
                    h = h[:, :max_tok]
                
                # specific cutting when using dependency information
                # here we cut on max words per sentence and nr of sentences.
                elif cut == "sent":
                    
                    if len(h.shape) == 2:
                        h = h[:, :max_sent]

                    elif len(h.shape) > 2:
                        h = h[:, :max_sent, :max_sent_tok]
                                    
            except ValueError as e:
                h = [s[k] for s in samples_data]
            
            except IndexError as e:
                h = h 


            #print(k, h, type(h), k2dtype.get(k, torch.long))
            unpacked_data[k] = to_tensor(h, dtype=k2dtype.get(k, torch.long))


        # return samples_data
        return Batch(
                        deepcopy(unpacked_data), 
                        tasks=self.tasks,
                        prediction_level=self.prediction_level,
                        length=len(unpacked_data["ids"])
                    )
    

    def __len__(self):
        return self.nr_samples


    @property
    def name(self):
        return self._name


    def __extract_sample_data(self, key):

        #samples = self.level_dfs["token"].loc[key,].groupby(self.sample_level+"_id")
        samples = self.data.loc[key,].groupby(self.sample_level+"_id")
        #samples = sorted(samples, key=lambda x:len(x), reverse=True)

        sample_data = []
        for i, sample in samples:

            sample_dict = {"ids": i}

            if self.prediction_level == "ac":
                acs_grouped = sample.groupby("ac_id")
                sample_dict.update({
                                    "lengths_tok":len(sample), 
                                    "lengths_seq":len(acs_grouped), 
                                    "lengths_seq_tok": [len(g) for i, g in acs_grouped],
                                    })
            else:
                sample_dict.update({"lengths_tok":len(sample)})

            if hasattr(self, "max_sent"):
                sent_grouped = sample.groupby("sentence_id")
                sample_dict.update({
                                    "lengths_sent": len(sent_grouped),
                                    "lengths_sent_tok": [len(g) for i, g in sent_grouped]
                                    })

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
                                            shape=self.nr_samples + 1
                                            )


    def __get_text(self, sample):
        sample_text = {"text": ""}

        if self.prediction_level == "ac":
            acs = sample.groupby(f"ac_id")
            text = []
            for _, ac in acs:
                text.append(ac["text"].to_numpy())

            sample_text["text"] = np.array(text)
        else:
            sample_text["text"] = sample["text"].to_numpy()
        
        return sample_text


    def __get_shape(self, sentence:bool=False, ac:bool=False, token:bool=False, char:bool=False, feature_dim:int=False):

        shape = []

        if sentence:
            shape.append(self.max_sent)

        if ac:
            shape.append(self.max_seq)

        if token and ac:
            shape.append(self.max_seq_tok)
        elif token:
            shape.append(self.max_tok)

        
        if char:
            shape.append(self.encoders["chars"].max_word_length)

        if feature_dim:
            shape.append(feature_dim)
        
        return tuple(shape)


    def __get_labels(self, sample):

        sample_labels = {}
        for task in self.all_tasks:

            if self.prediction_level == "ac":
                task_matrix = np.zeros(self.max_seq)

                ac_i = 0
                acs = sample.groupby(f"ac_id")
                for _, ac in acs:
                    task_matrix[ac_i] = np.nanmax(ac[task].to_numpy())
                    ac_i += 1
            else:
                task_matrix = np.zeros(self.max_tok)
                task_matrix[:sample.shape[0]]  = sample[task].to_numpy()

            sample_labels[task] = task_matrix
        
        return sample_labels
    

    def __get_encs(self, sample):

        sample_encs = {}
        deps_done = False

        for enc in self.encodings:

            # for dependency information we need to perserve the sentences 
            #
            if self.sample_level != "sentence" and enc in ["deprel", "dephead"]:
                
                if deps_done:
                    continue
                
                deps_done = True

                deprel_m = np.zeros((self.max_sent, self.max_sent_tok))      
                dephead_m = np.zeros((self.max_sent, self.max_sent_tok))      
                sent2root = np.zeros(self.max_sent)

                sentences  = sample.groupby("sentence_id")

                ac_i = 0

                #create a mask
                if self.prediction_level == "ac":
                    
                    #information about which ac belong in which sentence
                    ac2sentence = np.zeros(self.max_seq)

                    # given information from ac2sentence, we can get the mask for a sentnece for a particular ac
                    # so we know what tokens belong to ac and not
                    sentence_ac_mask = np.zeros((self.max_seq, self.max_sent_tok))


                for sent_i, (sent_id, sent_df) in enumerate(sentences):


                    deprel_m[sent_i][:sent_df.shape[0]] = sent_df["deprel"].to_numpy()
                    dephead_m[sent_i][:sent_df.shape[0]] = sent_df["dephead"].to_numpy()

                    root_id = self.encode_list(["root"], "deprel")[0]
                    root_idx = int(np.where(sent_df["deprel"].to_numpy() == root_id)[0])
                    sent2root[sent_i] = root_idx
                                    
                    #create a mask
                    if self.prediction_level == "ac":
                        acs = sent_df.groupby("ac_id")

                        for ac_id, ac_df in acs:
                            ac2sentence[ac_i] = sent_i

                            ac_ids = sent_df["ac_id"].to_numpy()
                            mask = ac_ids == ac_id
                            sentence_ac_mask[ac_i][:sent_df.shape[0]] = mask.astype(np.int8)
                            ac_i += 1


                if self.prediction_level == "ac":
                    sample_encs["ac2sentence"] = ac2sentence
                    sample_encs["sent_ac_mask"] = sentence_ac_mask


                sample_encs["sent2root"] = sent2root
                sample_encs["deprel"] = deprel_m
                sample_encs["dephead"] = dephead_m


            else:
                inc_ac = self.prediction_level == "ac" and not self.tokens_per_sample
                shape = self.__get_shape(
                                        ac=inc_ac,
                                        token=enc in set(["words", "bert_encs", "chars", "pos","deprel", "dephead" ]),
                                        char=enc == "chars",
                                        feature_dim=None
                                        )
                sample_m = np.zeros(shape)

                if inc_ac:

                    acs = sample.groupby("ac_id")
                    for ac_i,(_, ac ) in enumerate(acs):                        
                        sample_m[ac_i][:ac.shape[0]] = np.stack(ac[enc].to_numpy())

                else:
                    sample_m[:sample.shape[0]] = np.stack(sample[enc].to_numpy())
                
                sample_encs[enc] = sample_m
        


        return sample_encs


    def __get_feature_data(self, sample):
        
        mask_dict = {}
        feature_dict = {}
        masks_not_added = True

        if self.prediction_level == "ac":
            mask_dict["am_token_mask"] = np.zeros((self.max_seq, self.max_seq_tok))
            mask_dict["ac_token_mask"] = np.zeros((self.max_seq, self.max_seq_tok))
            mask_dict["ac_mask"] = np.zeros(self.max_seq)
    
        if self.tokens_per_sample:
            mask_dict["token_mask"] = np.zeros(self.max_tok)

        if self.prediction_level == "token":
            mask_dict["token_mask"] = np.zeros(self.max_tok)

    
        for feature, fm in self.feature2model.items():
            
            if hotam.preprocessing.settings["STORE_FEATURES"] and hasattr(fm, "_store_features"):
                self.__feature_store_setup(fm)

            if self.prediction_level == "ac":
                if self.tokens_per_sample:
                    if fm.level == "doc":
                        inc_ac = True
                    else:
                        inc_ac = False
                else:
                    inc_ac = True
            else:
                inc_ac = False

            shape = self.__get_shape(
                                    ac=inc_ac,
                                    token=fm.level == "word",
                                    #char=enc == "chars",
                                    feature_dim=fm.feature_dim
                                    )  

            feature_matrix = np.zeros(shape)
            sample_length = sample.shape[0]

            
            if self.prediction_level == "ac" and inc_ac:
                    
                feature_matrix, am_mask, ac_mask = self.__extract_ADU_features(fm, sample, sample_shape=feature_matrix.shape)

                if feature in self.__word_features and not np.sum(mask_dict["ac_token_mask"]):
                    mask_dict["am_token_mask"] = am_mask
                    mask_dict["ac_token_mask"] = ac_mask

                    if not sum(mask_dict["ac_mask"]):
                        mask_dict["ac_mask"] = np.max(ac_mask, axis=-1)
                else:
                    mask_dict["ac_mask"] = ac_mask
           
            else:
                
                # context is for embeddings such as Bert and Flair where the word embeddings are dependent on the surrounding words
                # so for these types we need to extract the embeddings per context. E.g. if we have a document and want Flair embeddings
                # we first divide the document up in sentences, extract the embeddigns and the put them back into the 
                # ducument shape.
                # Have chosen to not extract flair embeddings with context larger than "sentence".
                if fm.context and fm.context != self.sample_level:

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
                

                mask_dict["token_mask"][:sample_length] = np.ones(sample_length)


            if fm.group not in feature_dict:
                feature_dict[fm.group] = []

            feature_dict[fm.group].append(feature_matrix)

        feature_dict = {k: np.concatenate(v, axis=-1) if len(v) > 1 else v[0] for k,v in feature_dict.items()}
        feature_dict.update(mask_dict)

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
                    ac_mask_matrix[ac_i] = 1
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
            #nr_sentence = len(self.level_dfs["token"]["sentence_id"].unique())
            nr_sentence = len(self.data["sentence_id"].unique())
            shape = (nr_sentence, self._get_nr_tokens("sentence"), fm.feature_dim)
            feature_name = fm.name
        
        elif self.prediction_level == "ac":
            #body, suffix = self._feature2memmap[feature].rsplit(".",1)
            #self._feature2memmap[feature] = body + "_ac_" + suffix
            feature_name = fm.name + "_ac"

            #nr_acs = int(self.level_dfs["token"]["ac_id"].max())
            nr_acs = int(self.data["ac_id"].max())
            shape = (nr_acs, fm.feature_dim)

        else:
            raise NotImplementedError("Not supported yet")
            
        fm._init_feature_save(
                                dataset_name = self.name,
                                feature_name = feature_name,
                                shape = shape,
                                dtype = fm.dtype,
                                )


    def info(self):
        doc = f"""
            Info:

            {self.about}

            Tasks: 
            {self.dataset_tasks}

            Task Labels:
            {self.dataset_task_labels}

            Source:
            {self.url}
            """
        print(doc)


    def stats(self, override=False):

        """

        Returns
        -------
        [type]
            [description]

        """

    
        if not self._setup_done:

            #nr samples
            nr_samples = len(self.data.groupby(self.dataset_level+"_id"))
            acs = self.data.groupby("ac_id")
      
            rows = []
            for task in self.dataset_tasks:
                task_label_counts = acs.first()[task].value_counts()

                total = 0
                for l, c in task_label_counts.items():
                    rows.append((task,l,c))
                    total += c
                rows.append((task,"TOTAL",total))
            
            print("------ DATASET STATS ------ \n")
            print(f"Number Samples ({self.dataset_level}):", nr_samples)
            print("Label stats:")
            df = pd.DataFrame(rows, columns=["task","label", "count"])
            df.index = df.pop("task")

            return df



        if hasattr(self, "_stats") and not override:
            return self._stats
        
        rows = []

        ac_df = self.data.groupby("ac_id").first()

        # we go through each of the different splits (e.g. each cross validation set. For non-cross validation there will only be one)
        for split_id, split_set in self.splits.items():

            #we got through every split type (train, test ,val)
            for split_type, ids in split_set.items():
                
                
                rows.append({
                                "type":self.sample_level,
                                "task":"",
                                "split":split_type,
                                "split_id":split_id,
                                "value": len(ids)

                            })

                for task in self.all_tasks:

                    if "seg" in task:
                        label_counts = dict(self.data[self.data[f"{self.sample_level}_id"].isin(ids)][task].value_counts())
                    else:
                        label_counts = dict(ac_df[ac_df[f"{self.sample_level}_id"].isin(ids)][task].value_counts())

                    #if self.prediction_level == "ac" and "None" in label_counts:

                    
                    label_counts.update({l:0 for l in self.task2labels[task] if l not in label_counts.keys()})
                    label_counts = dict(sorted(list(label_counts.items()), key=lambda x:x[0]))

                    if "None" in label_counts and task == "ac":
                        label_counts.pop("None")

                    for l,nr in label_counts.items():
            
                        rows.append({
                                        "type":l,
                                        "task":task,
                                        "split":split_type,
                                        "split_id":split_id,
                                        "value": nr
                                    })

        self._stats = pd.DataFrame(rows)
        #df.index = df.pop("type")

        return self._stats


    @one_tqdm(desc="Saving Preprocessed Dataset")
    def save(self, pkl_path:str):
        """saves the preprocessed dataset, save dataset_level

        Parameters
        ----------
        pkl_path : str
            string path
        """
        pickle_data([self.data, self.dataset_level, self.dataset_tasks], pkl_path)


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

        self.data, self.dataset_level, self.dataset_tasks = load_pickle_data(pkl_path)
        # self.stack_level_data = {l:[] for l in self.level_dfs.keys()}
        self._data_stack = []


    @one_tqdm(desc="Save Encoded State")
    def __save_enc_state(self):

        state_dict = {}
        # state_dict["data"] = self.level_dfs["token"].to_dict()
        state_dict["data"] = self.data.to_dict()
        state_dict["all_tasks"] = self.all_tasks
        state_dict["subtasks"] = self.subtasks
        state_dict["main_tasks"] = self.main_tasks
        state_dict["task2subtasks"] = self.task2subtasks
        state_dict["task2labels"] = self.task2labels
        state_dict["stats"] =  self._stats.to_dict()


        with open(self._enc_file_name,"w") as f:
            json.dump(state_dict, f)


    @one_tqdm(desc="Load Encoded State")
    def __load_enc_state(self):
        with open(self._enc_file_name,"r") as f:
            state_dict = json.load(f)

        # self.level_dfs["token"] = pd.DataFrame(state_dict["data"])
        self.data = pd.DataFrame(state_dict["data"])
        self.all_tasks = state_dict["all_tasks"]
        self.subtasks = state_dict["subtasks"]
        self.task2subtasks = state_dict["task2subtasks"]
        self.task2labels = state_dict["task2labels"]
        self.main_tasks = state_dict["main_tasks"]
        self._stats = pd.DataFrame(state_dict["stats"])

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
                #subtasks_set.update(subtasks)
                all_task.update(subtasks)
                break
        
            self.task2subtasks[task] = subtasks

            #subtasks_set.update(subtasks)
            all_task.update(subtasks)

        
        self.all_tasks = sorted(list(all_task))
        self.subtasks = [t for t in all_task if "_" not in t]


    @one_tqdm(desc="Getting task labels")
    def __get_task2labels(self):
        self.task2labels = {}
        for task in self.all_tasks:

            # if task == "relation" and self.prediction_level == "ac":
            #     self.task2labels[task] = range(self.max_relation+1)
            # else:
            #     # self.task2labels[task] = sorted(list(self.level_dfs["token"][task].unique()))

            # for i, row in self.data.iterrows():
            #     if type(row["seg"]) == type(np.nan):
            #         print(row)

            # print(task, self.data[task].unique().tolist())
            self.task2labels[task] = sorted(self.data[task].unique().tolist())

            if isinstance(self.task2labels[task][0], (np.int, np.int64, np.int32, np.int16)):
                self.task2labels[task] = [int(i) for i in self.task2labels[task]]

        # if we only predict on acs we dont need "None" label
        if self.prediction_level == "ac" and "ac" in self.all_tasks:
            self.task2labels["ac"].remove("None")
        
        
    @one_tqdm(desc="Reformating Labels")
    def __fuse_subtasks(self):

        for task in self.tasks:
            subtasks = task.split("_")
            
            if len(subtasks) <= 1:
                continue

            # subtask_labels  = self.level_dfs["token"][subtasks].apply(lambda row: '_'.join([str(x) for x in row]), axis=1)
            # self.level_dfs["token"][task] = subtask_labels
            subtask_labels  = self.data[subtasks].apply(lambda row: '_'.join([str(x) for x in row]), axis=1)
            self.data[task] = subtask_labels


    def _get_nr_tokens(self, level):
        """
        Gets the max number of words for each of the levels in the heirarchy. E.g. max words in documents, 
        sentences, paragraphs.

        """
        # return max(len(g) for i,g in self.level_dfs["token"].groupby(level+"_id"))
        return max(len(g) for i,g in self.data.groupby(level+"_id"))
  

    def _get_max_nr_seq(self, level):
        # samples = self.level_dfs["token"].groupby(self.sample_level+"_id")
        samples = self.data.groupby(self.sample_level+"_id")
        return max(len(sample.groupby(level+"_id")) for i, sample in samples)


    @one_tqdm(desc="Removing Duplicates")
    def remove_duplicates(self):
        """
        Removes duplicate on the dataset level based on the text and also updates the splits to match the pruned dataset.

        text filed which is used to check duplicates is the original text.

        """
        raise NotImplementedError(" TO BE IMPLEMENTED")


        # def __get_duplicate_set(df):

        #     # TODO: solve this with groupby isntead???
        #     duplicate_mask_all = df.duplicated(subset="text",  keep=False).to_frame(name="bool")
        #     duplicate_ids_all = duplicate_mask_all[duplicate_mask_all["bool"]==True].index.values

        #     dup_ids = df.loc[duplicate_ids_all]["id"]
        #     dup_items = df.loc[duplicate_ids_all]["text"]
        #     dup_dict = dict(enumerate(set(dup_items)))
        #     dup_pairs = {i:[] for i in range(len(dup_dict))}
        #     for i, item in dup_dict.items():
        #         for ID, item_q in zip(dup_ids, dup_items):
        #             if item == item_q:
        #                 dup_pairs[i].append(ID)

        #     return list(dup_pairs.values())

        # target_level = self.dataset_level
        # # df = self.level_dfs[target_level]
        # # df = self.data
        # duplicate_sets = __get_duplicate_set(df)
        # duplicate_ids = np.array([i for duplicate_ids in duplicate_sets for i in  duplicate_ids[1:]])

        # df = df[~df["id"].isin(duplicate_ids)]
        # df.reset_index(drop=True, inplace=True)
        # self.level_dfs[target_level] = df


        # for level, df in self.level_dfs.items():

        #     if level == target_level:
        #         continue

        #     df = df[~df[f"{target_level}_id"].isin(duplicate_ids)]
        #     df.reset_index(drop=True, inplace=True)
        #     self.level_dfs[level] = df

        
        # self.duplicate_ids = duplicate_ids

        # if hasattr(self, "splits"):
        #     self._update_splits()

        # logger.info(f"Removed {len(duplicate_ids)} duplicates from dataset. Duplicates: {duplicate_sets}")


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


    def __get_tree_data(self, example):

        acs_grouped = example.groupby("ac_id")
        relations = [ac_df["relation"].unique()[0] for ac_id, ac_df in acs_grouped]
        acs = [ac_df["ac"].unique()[0] for ac_id, ac_df in acs_grouped]
        tree_data = []
        
        if self.name.lower() == "pe":
            major_claim_idx = [i for i, (ac_id, ac_df) in enumerate(acs_grouped) if ac_df["ac"].unique()[0] == "MajorClaim"][0]

        for i, (ac_id, ac_df) in enumerate(acs_grouped):
            text = " ".join(ac_df["text"])
            ac = ac_df["ac"].unique()[0]

            if self._setup_done:
                if "ac" in self.all_tasks:
                    ac = self.encoders["ac"].decode(ac)

            ac = ac_df["ac"].unique()[0]

            relation = int(ac_df["relation"].unique()[0])
            relation = relation if self._setup_done else i + relation
            stance = ac_df["stance"].unique()[0]

            if ac == "MajorClaim" and self.name.lower() == "pe":
                relation = major_claim_idx
                stance = "Paraphrase"
     
            if ac == "Claim" and self.name.lower() == "pe":
                relation  = major_claim_idx

            tree_data.append({
                                'label': ac,
                                'link': relation,
                                'link_label': stance,
                                'text': text
                                })
        return tree_data
    

    def __get_span_data(self, example):
        text_span_data = []
        for i, row in example.iterrows():
            span_id = row["ac_id"] if not np.isnan(row["ac_id"]) else None
            label = row["ac"]

            if self._setup_done:
                if "ac" in self.all_tasks:
                    ac = self.encoders["ac"].decode(ac)

            score = 1.0 if label != None else 0.0
            
            text_span_data.append({
                                'token': row["text"],
                                'pred': {
                                            "span_id": span_id,
                                            "label": label,
                                            "score": score,
                                            },
                                'gold':{
                                            "span_id": span_id,
                                            "label": label,
                                            "score": score,
                                            },
                                }) 
        return text_span_data


    def example(self, sample_id="random", level="document", span_params:dict={}, tree_params:dict={}):
        

        if sample_id == "random":
            if self._setup_done:
                sample_id = random.choice(set(self.data.index.to_numpy()))

            else:
                sample_id = random.choice(self.data[level+"_id"].unique())

        
        if self._setup_done:
            example = self.data.loc[sample_id]
            can_do_tree = "relation" in self.all_tasks
            can_do_spans = "seg" in self.all_tasks or "ac" in self.all_tasks
            show_scores = "ac" in self.all_tasks
        else:
            example = self.data.loc[self.data[level+"_id"] == sample_id, :]
            can_do_tree = True
            can_do_spans = True
            show_scores = True

        if can_do_spans:
            text_span_data = self.__get_span_data(example)

            hot_text_args = dict(labels=self.dataset_task_labels["ac"], 
                                save_path="/tmp/hot_text.png", 
                                show_scores=show_scores,
                                show_gold=False,
                                height=600)
            hot_text_args.update(span_params)

            hot_text(text_span_data, **hot_text_args)
            display(Image(filename=hot_text_args["save_path"]))

        if can_do_tree:
            fig = hot_tree(self.__get_tree_data(example), **tree_params)
            fig.show()

        #fig.show()
        #if can_do_spans and can_do_tree:
        #display(Image(filename=hot_text_args["save_path"])), fig.show()

        # elif can_do_spans:
        #     display(Image(filename=hot_text_args["save_path"])),

        # elif can_do_tree:
        #      fig.show()


    def setup(  self,
                tasks:list,
                prediction_level:str,
                sample_level:str, 
                features:list=[],
                encodings:list=[],
                remove_duplicates:bool=False,
                tokens_per_sample:bool=False,
                override:bool=False,
                ):
        """prepares the data
        """

        if prediction_level == "ac" and [t for t in tasks if "seg" in t]:
            raise ValueError("If prediction level is ac you cannot have segmentation as a task")

        feature_levels = set([fm.level for fm in features])
        if "doc" in feature_levels and prediction_level == "token":
            raise ValueError("Having features on doc level is not supported when prediction level is on word level.")

        self._setup_done = True
        self.prediction_level = prediction_level
        self.sample_level = sample_level
        self.tokens_per_sample = tokens_per_sample
        self.main_tasks = tasks
        self.tasks = tasks
        self.encodings = encodings

        self.feature2model = {fm.name:fm for fm in features}
        self.features = list(self.feature2model.keys())
        self._feature_groups = set([fm.group for fm in features])
        self.feature2dim = {fm.name:fm.feature_dim for fm in features}
        self.feature2dim.update({
                                group:sum([fm.feature_dim for fm in features if fm.group == group]) 
                                for group in self._feature_groups
                                })

        #self.feature2dim["word_embs"] = sum(fm.feature_dim for fm in features if fm.level == "word")
        #self.feature2dim["doc_embs"] = sum(fm.feature_dim for fm in features if fm.level == "doc")
        self.__word_features = [fm.name for fm in self.feature2model.values() if fm.level == "word"]

        # to later now how we cut some of the padding for each batch
        if self.tokens_per_sample:
            self.__cutmap.update({k:"tok" for k in  encodings + ["word_embs"]})

        #create a hash encoding for the exp config
        #self._exp_hash = self.__create_exp_hash()
        self._enc_file_name = os.path.join("/tmp/", f"{'-'.join(self.tasks+self.encodings)+self.prediction_level}_enc.json")

        # if remove_duplicates:
        #     self.remove_duplicates()

        # #remove duplicates from splits
        # if remove_duplicates:
        #     if self.duplicate_ids.any():
        #         self.update_splits()


        if self.prediction_level == "token" or self.tokens_per_sample:
            self.max_tok = self._get_nr_tokens(self.sample_level)

        if self.prediction_level == "ac":
            self.max_seq = self._get_max_nr_seq("ac")
            self.max_seq_tok = self._get_nr_tokens(self.prediction_level)
        
        if self.sample_level != "sentence" and ("deprel" in self.encodings or  "dephead" in self.encodings):
            self.max_sent = self._get_max_nr_seq("sentence")
            self.max_sent_tok = self._get_nr_tokens("sentence")
            self.__cutmap["dephead"] = "sent"
            self.__cutmap["deprel"] = "sent"
        

        enc_data_exit = os.path.exists(self._enc_file_name)
        if enc_data_exit and not override:
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
            self.stats(override=True)
            self._encode_labels()
            self.__save_enc_state()
        
        
        # self.level_dfs["token"].index = self.level_dfs["token"][f"{sample_level}_id"].to_numpy() #.pop(f"{sample_level}_id")
        self.data.index = self.data[f"{sample_level}_id"].to_numpy() #.pop(f"{sample_level}_id")
        
        # self.nr_samples = len(self.level_dfs[self.sample_level].shape[0]
        self.nr_samples = len(self.data[self.sample_level+"_id"].unique())

        if self.sample_level != self.dataset_level:
            self._change_split_level()
        

        self.config = {
                        "prediction_level":prediction_level,
                        "sample_level": sample_level,
                        "tasks": tasks,
                        "subtasks": self.subtasks,
                        "encodings": encodings,
                        "features": self.features,
                        "remove_duplicates": remove_duplicates,
                        "task_labels": self.task2labels,
                        "tracked_sample_ids": {str(s): ids["val"][:20].tolist() for s, ids in self.splits.items()}
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
