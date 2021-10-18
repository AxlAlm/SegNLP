

#basics 
import numpy as np
import pandas as pd

#segnlp
from segnlp import utils


class LabelEncoder:


    def __init__(self, task_labels:dict):
        self.task_labels = task_labels
        self.__create_label_id_mappings()
        self.__create_decouple_mapping()

        self.seg_task = None
        if "seg" in self.task_labels.keys():
            self.seg_task = sorted([task for task in self.task_labels.keys() if "seg" in task], key = lambda x: len(x))[-1]
            self.seg_decoder = utils.BIODecoder()


    def validate(self, task :str, df:pd.DataFrame, level:str):
            
        if task == "seg":
            self.__decode_segs(df)

        if level == "token" and task != "seg":
            self.__ensure_homogeneous(task, df)

        if task == "link":
            self.__correct_links(df)

        subtasks = task.split("+")  
        if len(subtasks) > 1:
            self.decouple(
                            task = task,
                            subtasks = subtasks, 
                            df = df,
                            level = level
                            )
        return df
        

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
    

    def __decode_segs(self, df:pd.DataFrame):
    
        # we get the sample start indexes from sample lengths. We need this to tell de decoder where samples start
        sample_sizes = df.groupby("sample_id", sort = False).size().to_numpy()
        sample_end_idxs = np.cumsum(sample_sizes)
        sample_start_idxs = np.concatenate((np.zeros(1), sample_end_idxs))[:-1]

        df.loc[:, "seg_id"] = self.seg_decoder(
                                        df["seg"].to_numpy(), 
                                        sample_start_idxs=sample_start_idxs.astype(int)
                                        )


    def __correct_links(self, df:pd.DataFrame):
        """
        This function perform correction 3 mentioned in https://arxiv.org/pdf/1704.06104.pdf  (Appendix)
        Any link that is outside of the actuall text, e.g. when predicted link > max_idx, is set to predicted_link== max_idx
        """

        max_segs = df.groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()
        df.loc[:, "max_seg"] = np.repeat(max_segs, df.groupby(level=0, sort=False).size().to_numpy())

        above = df["link"] >= df["max_seg"]
        below = df["link"] < 0

        df.loc[above | below, "link"] = df.loc[above | below, "max_seg"] - 1
        df.pop("max_seg")


    def __ensure_homogeneous(self, task:str, df:pd.DataFrame):

        """
        ensures that the labels inside a segments are the same. For each segment we take the majority label 
        and use it for the whole span.
        """
        count_df = df.loc[:, ["seg_id", task]].value_counts(sort=False).to_frame()
        count_df.reset_index(inplace=True)
        count_df.rename(columns={0:"counts"}, inplace=True)
        count_df.drop_duplicates(subset=['seg_id'], inplace=True)

        print(count_df)
        print(count_df[task].to_numpy())
        print(lol)

        seg_lengths = df.groupby("seg_id", sort=False).size()
        most_common = np.repeat(count_df[task].to_numpy(), seg_lengths)

        df.loc[~df["seg_id"].isna(), task] = most_common


    def encode(self, task: str, df: pd.DataFrame, level: str = None):
        
        if task  == "link":

            #get the nr of segments per sample
            seg_per_sample = df.groupby("sample_id", sort=False)["seg_id"].nunique().to_numpy()

            # get the indexes of each segments per sample
            seg_sample_idxes = np.hstack([np.arange(a) for a in seg_per_sample])
            
            #repeat the indexes of each segment id per sample for number of tokens in each segment
            seg_sample_idxes_per_token = np.repeat(seg_sample_idxes, df.groupby("seg_id", sort=False).size().to_numpy())
            
            # make the link labels encoded to be pointing instead of being difference in nr segments
            enc_tok_links = seg_sample_idxes_per_token + df.loc[~df["seg_id"].isna(), "link"].to_numpy()

            df.loc[~df["seg_id"].isna(), "link"] = enc_tok_links

        else:
            encode_fn = lambda x: self.label2id[task].get(str(x), -1)
            df.loc[:,task] = df[task].apply(encode_fn)
        

    def decode(self):
        raise NotImplementedError

    
    def decouple(self, task:str, subtasks:list, df: pd.DataFrame, level: str):
        
        subtask_preds = df[task].apply(lambda x: np.array(self._decouple_mapping[task][x]))
        subtask_preds = np.stack(subtask_preds.to_numpy())


        if "seg" in subtasks:
            df.loc[:, "seg"] = subtask_preds[:, subtasks.index("seg")]
            self.__decode_segs(df)


        for i, subtask in enumerate(subtasks):

            if subtask == "seg":
                continue

            df.loc[:, subtask] = subtask_preds[:, i]

            if level == "token":
                self.__ensure_homogeneous(subtask, df)
                

            # for links we add the decoded label, i.e. value indicates number of segments plus or minus the head segment is positioned.
            # so we need to encode these links so they refer to the specific segment index in a sample. Do do this we need to 
            # first let the segmentation create segment ids. See below where this is done
            if subtask == "link":

                df.loc[:,subtask] = subtask_preds[:,i]
                self.encode(
                            task = "link", 
                            df = df, 
                            level = level
                            )

                self.__correct_links(df) 


