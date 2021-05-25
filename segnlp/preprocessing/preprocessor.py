#basic
from typing import Union, List, Dict, Tuple
import numpy as np
import os
import pandas as pd
import shutil



#segnlp
from segnlp import get_logger
from .encoder import Encoder
from .textprocessor import TextProcesser
from .labeler import Labeler
from .dataset_preprocessor import DataPreprocessor
from segnlp.utils import ModelInput

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
                tasks:list,
                subtasks:list,
                task_labels:dict,
                features:list = [],
                encodings:list = [],
                other_levels:list = [],
                ):
        super().__init__()
        self.tasks = tasks
        self.subtasks = subtasks
        self.all_tasks = sorted(set(tasks + self.subtasks))
        self.task_labels = task_labels

        self.prediction_level = prediction_level
        self.sample_level = sample_level
        self.input_level = input_level

        self.argumentative_markers = False 
        if "am" in other_levels:
            self.argumentative_markers = True
            self.am_extraction = "pre"

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

        self._need_deps = False
        if "deprel" in encodings:
            self._need_deps = True

        self.__need_bio = "seg" in subtasks
        self.__labeling = True

        self._init_preprocessor()
        self._init_encoder()
        self._init_DataPreprocessor()
        self._create_data_encoders()
        self._create_label_encoders()

        self._removed = 0
    
    @property
    def config(self):
        return {
                "tasks": self.tasks,
                "subtasks": self.subtasks,
                "all_tasks": self.all_tasks,
                "task_labels": self.task_labels,
                "prediction_level": self.prediction_level,
                "sample_level": self.sample_level,
                "input_level": self.input_level,
                "feature2dim": self.feature2dim,
                "encodings":self.encodings
                }


    def __call__(self, doc:dict) -> ModelInput: #docs:List[str],token_labels:List[List[dict]] = None, span_labels:List[dict] = None):
        Input = ModelInput()

        span_labels = doc.get("span_labels", None)
        token_labels = doc.get("token_labels", None)
        doc = doc["text"]

        doc_df = self._process_doc(doc)
        doc_id = int(doc_df[f"document_id"].to_numpy()[0])

        if self.input_level != self.sample_level:
            samples = doc_df.groupby(f"{self.sample_level}_id")
        else:
            samples = [(doc_id,doc_df)]

        for i, sample in samples:
            i -= self._removed
            
            #everything within this block should be sped up
            if self.__labeling:
                if span_labels:
                    sample = self._label_spans(sample, span_labels)

                if token_labels:
                    sample = self._label_tokens(sample, token_labels)
                
                if self.__need_bio:
                    sample = self._label_bios(sample)
                
                self.__fuse_subtasks(sample)
                self._encode_labels(sample)
            

            if self.argumentative_markers:
                sample = self._label_ams(sample, mode=self.am_extraction)
            
            if self.encodings:
                self._encode_data(sample)
                
            segs = sample.groupby("seg_id")
            seg_length = len(segs)
            
            if self.prediction_level == "seg" and seg_length == 0:
                #if we are prediction on Units but sample doesnt have any, we can skip it
                self._removed += 1
                continue
            
            #tokens
            Input.add("ids", i, None)
            Input.add("token_ids", sample.loc[:,"id"].to_numpy(), "token")
            Input.add("lengths", sample.shape[0], "token")
            Input.add("mask", np.ones(sample.shape[0], dtype=np.uint8), "token")
            
            seg_token_lengths = np.array([g.shape[0] for i, g in segs])
            Input.add("lengths", seg_length, "seg")
            Input.add("lengths_tok", seg_token_lengths, "seg")
            Input.add("mask", np.ones(seg_length, dtype=np.uint8), "seg")
            self.__get_seg_idxs(Input, sample)


            #spans
            if self.__labeling:
                spans_grouped = sample.groupby("span_id")
                length = len(spans_grouped)
                lengths = np.array([g.shape[0] for i, g in spans_grouped])

                Input.add("lengths", length, "span")
                Input.add("lengths_tok", lengths, "span")

                none_span_mask = (~np.isnan(spans_grouped.first()["seg_id"].to_numpy())).astype(np.uint8)
                Input.add("none_span_mask", none_span_mask, "span")


            #ams
            if self.argumentative_markers:
                ams = sample.groupby("am_id")
                # as length of <= 1 is problematic later when working with NNs
                # we set lenght to 1 as default, this should not change anything as 
                # representations for such AMs will remain 0
                Input.add("lengths", seg_length, "am")
                Input.add("lengths", seg_length, "adu")
                self.__get_am_seg_idxs(Input, sample)


            self.__get_text(Input, sample)
            self.__get_encs(Input, sample)
            self.__get_feature_data(Input, sample)

            if self.__labeling:
                self.__get_labels(Input, sample)


        return Input.to_numpy()
  

    def __get_text(self, Input:ModelInput, sample:pd.DataFrame):

        # if self.prediction_level == "seg":
        #     segs = sample.groupby(f"seg_id")

        #     text = []
        #     for _, seg in segs:
        #         text.append(seg["text"].to_numpy().tolist())
            
        #     print(text)
        #     text = string_pad(text, dtype="U30")
        #     print(text.shape)
        #     Input.add("text", text, "seg")
        #     #sample_text["text"] = np.array(text)
        # else:
            #Input.add("text", sample["text"].to_numpy().astype("S"))
        Input.add("text", sample["text"].to_numpy().astype("U30"), "token")
            #sample_text["text"] = sample["text"].to_numpy()
        
        #return sample_text


    def __get_labels(self, Input:ModelInput, sample:pd.DataFrame):

        sample_labels = {}
        for task in self.all_tasks:

            segs = sample.groupby(f"seg_id")
            seg_task_matrix = np.zeros(len(segs))
            for i, (seg_id, seg) in enumerate(segs):
                seg_task_matrix[i] = np.nanmax(seg[task].to_numpy())

            #if self.prediction_level == "token":
            #task_matrix = np.zeros(len(sample.index))
            #task_matrix[:sample.shape[0]] = sample[task].to_numpy()

            Input.add(task, sample[task].to_numpy().astype(np.int), "token", pad_value=-1)
            Input.add(task, seg_task_matrix.astype(np.int), "seg", pad_value=-1)
        
        return sample_labels
    

    def __get_sample_dep_encs(self,  Input:ModelInput, sample:pd.DataFrame):

        sentences  = sample.groupby("sentence_id")

        sent_length = 0
        deprels = []
        depheads = []
        root_idx = -1
        for sent_id, sent_df in sentences:
            
            sent_deprels = sent_df["deprel"].to_numpy()
            sent_depheads = sent_df["dephead"].to_numpy() + sent_length

            sent_root_id = self.encode_list(["root"], "deprel")[0]
            sent_root_idx = int(np.where(sent_df["deprel"].to_numpy() == sent_root_id)[0])
            
            if sent_length == 0 and root_idx == -1:
                root_idx = sent_root_idx
                sent_length = sent_df.shape[0]
            else:
                sent_depheads[sent_root_idx] = sent_length-1
                sent_length += sent_df.shape[0]
      
            deprels.extend(sent_deprels)
            depheads.extend(sent_depheads)

        Input.add("root_idxs", root_idx, "token")
        Input.add("deprel", np.array(deprels, dtype=np.int), "token")
        Input.add("dephead", np.array(depheads, dtype=np.int), "token")
  

    def __get_encs(self, Input:ModelInput, sample:pd.DataFrame):

        deps_done = False
        for enc in self.encodings:

            #if self.sample_level != "sentence" and enc in ["deprel", "dephead"]:
            if enc in ["deprel", "dephead"]:
                if not deps_done:
                    self.__get_sample_dep_encs(Input=Input, sample=sample)
                    deps_done = True
            else:
                # if self.prediction_level == "seg" and not self.tokens_per_sample:

                #     segs = sample.groupby("seg_id")
                #     nr_tok_segs = max([len(seg) for seg in segs])
                #     seg_matrix  = np.zeros(len(segs), nr_tok_segs, dtype=np.int)
                #     for seg_i,(_, seg ) in enumerate(segs):                        
                #         sample_m[seg_i][:seg.shape[0]] = np.stack(seg[enc].to_numpy())

                #     Input.add(enc, seg_matrix, "seg")
                # else:
                Input.add(enc, np.stack(sample[enc].to_numpy()).astype(np.int), "token")
            
  
    def __get_feature_data(self, Input:ModelInput, sample:pd.DataFrame):
        
        feature_dict = {}
        sample_length = sample.shape[0]

        sample_length = sample.shape[0]

        for feature, fm in self.feature2model.items():
    
            if fm.level == "doc" and self.prediction_level == "seg":
                
                segs = sample.groupby("seg_id")
                feature_matrix = np.zeros((len(segs), fm.feature_dim))
                for i,(seg_id, seg_df) in enumerate(segs):
                    # sent.index = sent["id"]
                    data = sample[sample["seg_id"] == seg_id]

                    if self.argumentative_markers:
                        am = sample[sample["am_id"] == seg_id]
                        data = pd.concat((am,data))

                    #adu.index = adu.pop("seg_id")
                    feature_matrix[i] = fm.extract(data)


            elif fm.level == "word":
                # context is for embeddings such as Bert and Flair where the word embeddings are dependent on the surrounding words
                # so for these types we need to extract the embeddings per context. E.g. if we have a document and want Flair embeddings
                # we first divide the document up in sentences, extract the embeddigns and the put them bsegk into the 
                # ducument shape.
                # Have chosen to not extract flair embeddings with context larger than "sentence".
                if fm.context and fm.context != self.sample_level:

                    contexts = sample.groupby(f"{fm.context}_id")

                    sample_embs = []
                    for _, context_data in contexts:
                        sample_embs.extend(fm.extract(context_data)[:context_data.shape[0]])

                    feature_matrix = np.array(sample_embs)
            
                else:
                    #feature_matrix[:sample_length] = fm.extract(sample)[:sample_length]
                    feature_matrix = fm.extract(sample)[:sample_length]

            else:
                feature_matrix = fm.extract(sample)[:sample_length]


            if fm.group not in feature_dict:
                feature_dict[fm.group] = {
                                        "level": "seg" if fm.level == "doc" else "token",
                                        "data":[]
                                        }
            

            feature_dict[fm.group]["data"].append(feature_matrix)


        for group_name, group_dict in feature_dict.items():
            if len(group_dict["data"]) > 1:
                Input.add(group_name, np.concatenate(group_dict["data"], axis=-1), group_dict["level"])
            else:
                Input.add(group_name, group_dict["data"][0], group_dict["level"])


    def __get_seg_idxs(self, Input:ModelInput, sample:pd.DataFrame):

        am_spans = []
        seg_spans = []
        adu_spans = []

        segs = sample.groupby("seg_id")

        for seg_id, gdf in segs:
    
            seg_start = min(gdf[f"{self.sample_level}_token_id"])
            seg_end = max(gdf[f"{self.sample_level}_token_id"])
            seg_span = (seg_start, seg_end)

            seg_spans.append(seg_span)
        
        if not seg_spans:
            seg_spans = [(0,0)]

        #print(seg_spans)
        Input.add("span_idxs", np.array(seg_spans), "seg")


    def __get_am_seg_idxs(self, Input:ModelInput, sample:pd.DataFrame):
        """
        for each sample we get the segs of am, seg and the whole adu.
        if there is no am, we still add an am seg to keep the am and seg segs
        aligned. But we set the values to 0 and start the adu from the start of the seg instead
        of the am.

        """
        
        am_spans = []
        adu_spans = []

        segs = sample.groupby("seg_id")

        for seg_id, gdf in segs:
            
            am = sample[sample["am_id"]==seg_id]
            
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

            seg_start = min(gdf[f"{self.sample_level}_token_id"])
            seg_end = max(gdf[f"{self.sample_level}_token_id"])
            seg_span = (seg_start, seg_end)

            if has_am:
                adu_span = (am_start, seg_end)
            else:
                adu_span = (seg_start, seg_end)

            am_spans.append(am_span)
            adu_spans.append(adu_span)
        
        if not am_spans:
            am_spans = [(0,0)]

        if not adu_spans:
            adu_spans = [(0,0)]
        
        Input.add("span_idxs", np.array(am_spans), "am")
        Input.add("span_idxs", np.array(adu_spans), "adu")


    def __fuse_subtasks(self, df):

        for task in self.tasks:
            subtasks = task.split("+")
            
            if len(subtasks) <= 1:
                continue

            subtask_labels  = df[subtasks].apply(lambda row: '_'.join([str(x) for x in row]), axis=1)
            df[task] = subtask_labels


    def deactivate_labeling(self):
        self.__labeling = False


    def activate_labeling(self):
        self.__labeling = True


