#basic
from typing import Union, List, Dict, Tuple
import numpy as np
import os
import pandas as pd


#segnlp
from segnlp import get_logger
from segnlp.utils import Input


logger = get_logger("DocProcessor")


class DocProcessor:    


    def _process_doc(self, doc:dict) -> Input: #docs:List[str],token_labels:List[List[dict]] = None, span_labels:List[dict] = None):
        input = Input()

        span_labels = doc.get("span_labels", None)
        #token_labels = doc.get("token_labels", None)
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

                # if token_labels:
                #     sample = self._label_tokens(sample, token_labels)
                
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
            input.add("ids", i, None)
            input.add("token_ids", sample.loc[:,"id"].to_numpy(), "token")
            input.add("lengths", sample.shape[0], "token")
            input.add("mask", np.ones(sample.shape[0], dtype=np.uint8), "token")
            input.add("seg_id", sample["seg_id"].to_numpy(), "token")
            
            seg_token_lengths = np.array([g.shape[0] for i, g in segs])
            input.add("lengths", seg_length, "seg")
            input.add("lengths_tok", seg_token_lengths, "seg")
            input.add("mask", np.ones(seg_length, dtype=np.uint8), "seg")
            self.__get_seg_idxs(input, sample)


            #spans
            if self.__labeling:
                spans_grouped = sample.groupby("span_id")
                length = len(spans_grouped)
                lengths = np.array([g.shape[0] for i, g in spans_grouped])

                input.add("lengths", length, "span")
                input.add("lengths_tok", lengths, "span")

                none_span_mask = (~np.isnan(spans_grouped.first()["seg_id"].to_numpy())).astype(np.uint8)
                input.add("none_span_mask", none_span_mask, "span")


            #ams
            if self.argumentative_markers:
                ams = sample.groupby("am_id")
                # as length of <= 1 is problematic later when working with NNs
                # we set lenght to 1 as default, this should not change anything as 
                # representations for such AMs will remain 0
                input.add("lengths", seg_length, "am")
                input.add("lengths", seg_length, "adu")
                self.__get_am_seg_idxs(input, sample)


            self.__get_text(input, sample)
            self.__get_encs(input, sample)
            self.__get_feature_data(input, sample)

            if self.__labeling:
                self.__get_labels(input, sample)


        return input.to_numpy()
  

    def __get_text(self, input:Input, sample:pd.DataFrame):

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
        input.add("text", sample["text"].to_numpy().astype("U30"), "token")
            #sample_text["text"] = sample["text"].to_numpy()
        
        #return sample_text


    def __get_labels(self, input:Input, sample:pd.DataFrame):

        sample_labels = {}
        for task in self.all_tasks:

            segs = sample.groupby(f"seg_id")
            seg_task_matrix = np.zeros(len(segs))
            for i, (seg_id, seg) in enumerate(segs):
                seg_task_matrix[i] = np.nanmax(seg[task].to_numpy())

            #if self.prediction_level == "token":
            #task_matrix = np.zeros(len(sample.index))
            #task_matrix[:sample.shape[0]] = sample[task].to_numpy()

            input.add(task, sample[task].to_numpy().astype(np.int), "token", pad_value=-1)
            input.add(task, seg_task_matrix.astype(np.int), "seg", pad_value=-1)
        
        return sample_labels
    

    def __get_sample_dep_encs(self,  input:Input, sample:pd.DataFrame):

        sentences  = sample.groupby("sentence_id")

        sent_length = 0
        deprels = []
        depheads = []
        root_idx = -1
        for sent_id, sent_df in sentences:
            
            sent_deprels = sent_df["deprel"].to_numpy()
            sent_depheads = sent_df["dephead"].to_numpy() + sent_length

            sent_root_id = self.encode_list(["root"], "deprel")[0]
            sent_root_idx_match = np.where(sent_df["deprel"].to_numpy() == sent_root_id)

            # if we dont find a sentence root, we default the root to the first word
            if not sent_root_idx_match[0]:
                sent_root_idx = 0
            else:
                sent_root_idx = int(sent_root_idx_match[0])


            if sent_length == 0 and root_idx == -1:
                root_idx = sent_root_idx
                sent_length = sent_df.shape[0]
            else:
                sent_depheads[sent_root_idx] = sent_length-1
                sent_length += sent_df.shape[0]
      
            deprels.extend(sent_deprels)
            depheads.extend(sent_depheads)

        input.add("root_idxs", root_idx, "token")
        input.add("deprel", np.array(deprels, dtype=np.int), "token")
        input.add("dephead", np.array(depheads, dtype=np.int), "token")
  

    def __get_encs(self, input:Input, sample:pd.DataFrame):

        deps_done = False
        for enc in self.encodings:

            #if self.sample_level != "sentence" and enc in ["deprel", "dephead"]:
            if enc in ["deprel", "dephead"]:
                if not deps_done:
                    self.__get_sample_dep_encs(input=input, sample=sample)
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
                input.add(enc, np.stack(sample[enc].to_numpy()).astype(np.int), "token")
            
  
    def __get_feature_data(self, input:Input, sample:pd.DataFrame):
        
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
                input.add(group_name, np.concatenate(group_dict["data"], axis=-1), group_dict["level"])
            else:
                input.add(group_name, group_dict["data"][0], group_dict["level"])


    def __get_seg_idxs(self, input:Input, sample:pd.DataFrame):

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
        input.add("span_idxs", np.array(seg_spans), "seg")


    def __get_am_seg_idxs(self, input:Input, sample:pd.DataFrame):
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
        
        input.add("span_idxs", np.array(am_spans), "am")
        input.add("span_idxs", np.array(adu_spans), "adu")


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


