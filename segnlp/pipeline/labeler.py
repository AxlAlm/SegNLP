
#basics
import re
import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm


#segnlp
from segnlp import utils
from segnlp import get_logger
from segnlp.resources.am import find_am

logger = get_logger("LABELER")

class Labeler:


    def _encode_labels(self, df):

        for task in self.all_tasks:
            df = self.label_encoder.encode( 
                                        task = task,
                                        df = df
                                        )
        
        return df

    def _decode_labels(self, df: pd.DataFrame):
        raise NotImplementedError


    def _label_spans(self, df:pd.DataFrame, span_labels:dict):

        def label_f(row, span_labels):
            return span_labels.get(int(row["char_end"]),{})

        df = pd.concat([df,df.apply(label_f, axis=1, result_type="expand", args=(span_labels,))], axis=1)
        return df


    def _fuse_subtasks(self, df: pd.DataFrame):

        for task in self.tasks:
            subtasks = task.split("+")
            
            if len(subtasks) <= 1:
                continue

            subtask_labels  = df[subtasks].apply(lambda row: '_'.join([str(x) for x in row]), axis=1)
            df[task] = subtask_labels


    def _label_bios(self, df: pd.DataFrame):
        df["seg"] = "O"
        segs = df.groupby("seg_id", sort=False)
        for seg_id, seg_df in segs:
            df.loc[seg_df.index, "seg"] = ["B"] +  (["I"] * (seg_df.shape[0]-1))
        return df


    def _label_ams(self, df: pd.DataFrame, mode="pre"):

        if mode == "pre":
            df = self.__ams_as_pre(df)

        elif mode == "from_list":
            df = self.__ams_from_list(df)

        elif mode == "list_pre":
            raise NotImplementedError()

        return df


    def __ams_as_pre(self, df: pd.DataFrame):
        df["am_id"] = np.nan
        groups = df.groupby("sentence_id", sort=False)

        for sent_id, sent_df in groups:
            
            acs = sent_df.groupby("seg_id", sort=False)
            prev_ac_end = 0
            for ac_id, ac_df in acs:
                
                ac_start = min(ac_df["char_start"])
                ac_end = max(ac_df["char_end"])

                # more than previous end of ac and less than ac start
                cond1 = sent_df["char_start"] >= prev_ac_end 
                cond2 = sent_df["char_start"] < ac_start
                idxs = sent_df[cond1 & cond2].index

                #set the id to ac_id
                df.loc[idxs,"am_id"] = ac_id

                # text = " ".join(df.loc[idxs, "text"].to_numpy())
                # if text:
                #     with open("/tmp/pe_ams.txt","a") as f:
                #         f.write(text+"\n")

                prev_ac_end = ac_end

        return df


    def __ams_from_list(self, df: pd.DataFrame):

        df["am_id"] = np.nan
        groups = df.groupby("sentence_id", sort=False)

        for sent_id, sent_df in groups:
            
            acs = sent_df.groupby("seg_id", sort=False)

            for ac_id, ac_df in acs:

                tokens = ac_df["text"].to_list()
                am, am_indexes = find_am(tokens)

                if not am:
                    continue

                idx = ac_df.iloc[am_indexes].index
                df.loc[idx, "am_id"] = ac_id
                df.loc[idx, "ac_id"] = None

        return df
