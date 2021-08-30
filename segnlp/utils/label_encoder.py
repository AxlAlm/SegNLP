

#basics 
from logging import warn
import numpy as np
import pandas as pd


class LabelEncoder:


    def __init__(self, task_labels:dict):
        self.task_labels = task_labels
        self.__create_label_id_mappings()
        self.__create_decouple_mapping()


    def __create_decouple_mapping(self):
        
        decouplers = {}
        for task in self.task_labels:

            if "+" not in task:
                continue
            
            decouplers[task] = {}

            labels = list(self.label2id[task].keys())
            subtasks = task.split("+")

            for i,label in enumerate(labels):

                sublabels = label.split("_")

                encode_fn = lambda x,y: self.label2id[x].get(str(y), -1)

                decouplers[task][i] = [encode_fn(st, sl) if st != "link" else int(sl) for st, sl in zip(subtasks, sublabels)]

        self._decouple_mapping = decouplers
    

    def __create_label_id_mappings(self):
        
        self.label2id = {}
        self.id2label = {}

        for task in self.task_labels:
            
            if task == "link":
                continue
            
            # if "seg" in task:
            #     if  "O" not in self.task_labels[task][0]:
            #         raise Warning(f"Label at position 0 is not / does not contain 'O' for {task}: {self.task_labels[task]}")
            # else:
            #     if self.prediction_level == "token":
            #         if "None" not in self.task_labels[task][0]:
            #             raise Warning(f"Label at position 0 is not / does not contain 'None' for {task}: {self.task_labels[task]}")

            self.id2label[task] = dict(enumerate(self.task_labels[task]))
            self.label2id[task] = {l:i for i,l in self.id2label[task].items()}
    

    def encode(self, task: str, df: pd.DataFrame, level: str = None):
        
        if task in ["link", "T-link"]:
            
            key = "T-seg_id" if level == "seg" else "seg_id"

            # linkis = df.groupby(key, sort=False).first()["link"].to_numpy()
            # linkis_ENC = np.arange(len(linkis)) + linkis

            #get the nr of segments per sample
            seg_per_sample = df.groupby("sample_id", sort=False)[key].nunique().to_numpy()

            # get the indexes of each segments per sample
            seg_sample_idxes = np.hstack([np.arange(a) for a in seg_per_sample])
            
            #repeat the indexes of each segment id per sample for number of tokens in each segment
            seg_sample_idxes_per_token = np.repeat(seg_sample_idxes, df.groupby(key, sort=False).size().to_numpy())
            
            # make the link labels encoded to be pointing instead of being difference in nr segments
            enc_tok_links = seg_sample_idxes_per_token + df.loc[~df[key].isna(), "link"].to_numpy()

            df.loc[~df[key].isna(), "link"] = enc_tok_links

            #print(linkis_ENC, df.groupby(key, sort=False).first()["link"].to_numpy())
            #assert np.array_equal(linkis_ENC, df.groupby(key, sort=False).first()["link"].to_numpy())

        else:
            encode_fn = lambda x: self.label2id[task].get(str(x), -1)
            df[task].apply(encode_fn)
        
        return df


    def decode(self):
        raise NotImplementedError

    
    def decouple(self, task:str, subtasks: list, df: pd.DataFrame, level: str):

        # if our task is complexed, e.g. "seg+label". We decouple the label ids for "seg+label"
        # so we get the labels for Seg and for Label

        subtask_preds = df[task].apply(lambda x: np.array(self.label_decoupler[task][x]))
        subtask_preds = np.stack(subtask_preds.to_numpy())

        for i, subtask in enumerate(subtasks):

            # for links we add the decoded label, i.e. value indicates number of segments plus or minus the head segment is positioned.
            # so we need to encode these links so they refer to the specific segment index in a sample. Do do this we need to 
            # first let the segmentation create segment ids. See below where this is done
            if "link" == subtask:
                df = self.encode(
                            task = task, 
                            df = df, 
                            level = level
                            )
            else:
                df[subtask] = subtask_preds[:,i]


        return df