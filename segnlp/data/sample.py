
# basic
from typing import Generator, Sequence, List, Tuple
import pandas as pd
import numpy as np
from collections import Counter
from copy import deepcopy

# segnlp
from segnlp.utils import BIODecoder


# from .label_encoder import LabelEncoder
# from .array import ensure_numpy
# from .array import ensure_list
# from .array import create_mask
# from .array import np_cumsum_zero
# from .overlap import find_overlap
# from .misc import timer



class Sample:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self._size : int = len(df)
    

    def __len__(self) -> int:
        return self._size


    def __repr__(self) -> str:
        return " ".join(self.df["str"])


    def copy(self, clean_labels:bool =  False) -> "Sample":
        df = self.df.copy(deep = True)
        
        for task in self.__task_labels:
            df[task] = -1

        df["seg_id"] = -1
        df["target_id"] = -1

        # __init__ new
        copy_sample = Sample(df)
        
        # copy over some needed vars
        for var in ["__task_labels", "__decouple_mapping", "__id2label", "__label2id"]:
            if not hasattr(self, var):
                continue
            
            setattr(copy_sample, var, deepcopy(getattr(self, var)))


        return copy_sample


    def length(self) -> int:
        return self._size
    

    def pairs(self):
        pass

    
    def n_pairs(self) -> int:
        return len(self.pairs())


    def segs(self) -> Generator:
        for i, seg_df in self.df.groupby("seg_id", sort=False):

            if i == -1:
                continue

            yield i, seg_df


    def seg_ids(self) -> List[int]:
        return [i for i in self.segs()]


    def n_segs(self) -> int:
        return len(self.segs())
    
    
    def seg_lengths(self) -> int:
        return len(self.segs())
    

    def seg_span_idxs(self) -> List[Tuple[int, int]]:
        return [(seg_df.first()["id"], seg_df.last()["id"]) for i, seg_df in self.segs()]


    def split(self, level:str) -> List["Sample"]:
        return [Sample(df) for _, df in self.df.groupby(f"{level}_id", sort = False)]


    def get(self, level:str, key:str):

        if key in self.df.columns:
            return self.df[key].to_numpy()

        if key == "length":

            if level == "token":
                return len(self)
            
            if level == "seg":
                return self.n_segs()

            if level == "pair":
                return self.n_pairs()


        raise KeyError(f"Could not find values for {key} as level {level}")


    def add_preds(self):
        pass


    def add_segs(self, segs: Sequence):

        if "+" in self.seg_task:
            self.__decouple(segs)

            return 

        self.df["seg"] = segs
        self.__decode_segs()


    def add_links(self, links: Sequence) -> None:
        self.df["link"] = links
        self.__ensure_possible_links()


    def add_labels(self, labels : Sequence) -> None:
        self.df["label"] = labels
        self.__ensure_homogeneous("label")


    def add_link_labels(self, link_label: Sequence) -> None:
        self.df["link_label"] = link_label
        self.__ensure_homogeneous


    def __label_spans(self, span_labels:dict):

        def label_f(row, span_labels):
            return span_labels.get(int(row["char_end"]),{})

        self.df = pd.concat([self.df, self.df.apply(label_f, axis=1, result_type="expand", args=(span_labels,))], axis=1)
        #self.df.astype({'seg_id': 'float64'})


    def __fuse_tasks(self):

        for task in self.__task_labels:
            subtasks = task.split("+")
            
            if len(subtasks) <= 1:
                continue
            
            if "seg" in task:
                subtask_labels  = self.df[subtasks].apply(lambda row: '_'.join([str(x) for x in row if x is not "None"]), axis=1)
            else:
                subtask_labels  = self.df[subtasks].apply(lambda row: '_'.join([str(x) for x in row]), axis=1)

            self.df[task] = subtask_labels
        

    def __label_segs(self):
        self.df["seg"] = "O"
        segs = self.segs()
        for _, seg_df in segs:
            self.df.loc[seg_df.index, "seg"] = ["B"] +  (["I"] * (seg_df.shape[0]-1))


    def __encode(self, task):
        encode_fn = lambda x: self.__label2id[task].get(str(x), -1)
        self.df.loc[:,task] = self.df[task].apply(encode_fn)


    def __encode_tasks(self):

        for task in self.__task_labels:

            if task == "link":
                continue

            self.__encode(task)


    def __set_seg_target_id(self):
        pass


    def add_span_labels(self, span_labels:dict, task_labels:dict):

        self.__task_labels = task_labels
        self.__create_label_id_mappings()
        self.__create_decouple_mapping()
        self.__label_spans(span_labels)

        self.seg_task = None
        if "seg" in self.__task_labels:
            self.__label_segs()
            self.seg_task = sorted([task for task in self.__task_labels.keys() if "seg" in task], key = lambda x: len(x))[-1]
            self.seg_decoder = BIODecoder()
        
        self.__fuse_tasks()
        self.__encode_tasks()


    def __ensure_possible_links(self) -> None:
        """
        This function perform correction 3 mentioned in https://arxiv.org/pdf/1704.06104.pdf  (Appendix)
        Any link that is outside of the actuall text, e.g. when predicted link > max segs, is set to predicted_link == max_idx -1
        """

        self.df.loc[:, "max_seg"] = self.n_segs()

        above = self.df["link"] >= self.df["max_seg"]
        below = self.df["link"] < 0

        self.df.loc[above | below, "link"] = self.df.loc[above | below, "max_seg"] - 1
        self.df.pop("max_seg")


    def __ensure_homogeneous(self, task:str):
        """
        ensures that the labels inside a segments are the same. For each segment we take the majority label 
        and use it for the whole span.
        """
        most_common_label = [Counter(seg_df[task]).most_common(0)[0] for i, seg_df in self.segs()]
        most_common = np.repeat(most_common_label, self.seg_lengths())
        self.df.loc[self.seg_ids(), task] = most_common


    def __decode_segs(self):
        self.df.loc[:, "seg_id"] = self.seg_decoder(self.df["seg"].to_numpy())


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
        df["adu_id"] = df["seg_id"].to_numpy()
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
                df.loc[idxs,["am_id", "adu_id"]] = ac_id

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


    def __create_decouple_mapping(self):
        
        decouplers = {}
        for task in self.__task_labels:

            if "+" not in task:
                continue
            
            decouplers[task] = {}

            labels = list(self.__label2id[task].keys())
            subtasks = task.split("+")

            for i,label in enumerate(labels):
                    

                if "seg" in task and label == "O":
                    sublabels = ["O"] + (["None"]* (len(subtasks) -1))
                else:
                    sublabels = label.split("_")

                encode_fn = lambda task, label: self.__label2id[task].get(label, -1)
                decouplers[task][i] = [encode_fn(st, sl) for st, sl in zip(subtasks, sublabels)]

        self.__decouple_mapping = decouplers


    def __create_label_id_mappings(self):
        
        self.__label2id = {}
        self.__id2label = {}

        for task in self.__task_labels:
            
            if task == "link":
                continue
            
            # if "seg" in task:
            #     if  "O" not in self.task_labels[task][0]:
            #         raise Warning(f"Label at position 0 is not / does not contain 'O' for {task}: {self.task_labels[task]}")
            # else:
            #     if self.prediction_level == "token":
            #         if "None" not in self.task_labels[task][0]:
            #             raise Warning(f"Label at position 0 is not / does not contain 'None' for {task}: {self.task_labels[task]}")

            self.__id2label[task] = dict(enumerate(self.__task_labels[task]))
            self.__label2id[task] = {l:i for i,l in self.__id2label[task].items()}
    

    def overlap(self, sample:"Sample"):
        for i, seg_df in self.segs():

            for i2, seg_df2 in sample.segs():

                ratio = len(set(seg_df["str"]).intersection(set(seg_df2["str"]))) / max(len(seg_df), len(seg_df2))

                yield i, i2, ratio



def _overlap(j:int, target_seg_df: pd.DataFrame, pdf: pd.DataFrame) -> np.ndarray:

    # target segment id
    #j = target_seg_df["seg_id"].to_list()[0]

    # predicted segments in the sample target df
    pred_seg_ids = target_seg_df["PRED-seg_id"].dropna().to_list()

    if not pred_seg_ids:
        return np.array([None, None, None])
    
    #best pred segment id
    i = Counter(pred_seg_ids).most_common(1)[0][0]

    #get token indexes
    ps_token_ids = set(pdf.loc[[i], "token_id"]) #slowest part
    ts_token_ids = set(target_seg_df["token_id"])

    #calculate the overlap
    overlap_ratio = len(ts_token_ids.intersection(ps_token_ids)) / max(len(ps_token_ids), len(ts_token_ids))

    return np.array([i, j, overlap_ratio])


def find_overlap(pred_df : pd.DataFrame, target_df : pd.DataFrame) -> Tuple[dict]:

    # Create a new dfs which contain the information we need
    pdf = pd.DataFrame({
                        "token_id": pred_df["id"].to_numpy()
                        }, 
                        index = pred_df["seg_id"].to_numpy(),
                        )
    pdf = pdf[~pdf.index.isna()]

    tdf = pd.DataFrame({
                        "token_id": target_df["id"].to_numpy(),
                        "seg_id": target_df["seg_id"].to_numpy(),
                        "PRED-seg_id": pred_df["seg_id"].to_numpy(),
                        }, 
                        index = target_df["seg_id"].to_numpy(),
                        )
    
    tdf = tdf[~tdf.index.isna()]
    
    if tdf.shape[0] == 0:
        return {}, {}, {}, {}

    # extract information about overlaps
    overlap_info = np.vstack([ _overlap(j, tsdf, pdf) for j, tsdf in tdf.groupby("seg_id", sort=False)])    
  

    # we then filter out all Nones, and filter out all cases where j match with more than one i.
    # i.e. for each j we only selec the best i 
    df = pd.DataFrame(overlap_info, columns = ["i", "j", "ratio"])
    df = df.dropna()
    df = df.sort_values("ratio")
    top_matches = df.groupby("j", sort = False).first()

    i  = top_matches["i"].to_numpy(int)
    j = top_matches.index.to_numpy(int)
    ratio = top_matches["ratio"].to_numpy(float)

    i2ratio = dict(zip(i, ratio))
    j2ratio = dict(zip(j, ratio))
    i2j = dict(zip(i, j))
    j2i = dict(zip(j, i))

    return i2ratio, j2ratio, i2j, j2i










    # def __correct_links(self, df:pd.DataFrame):
    #     """
    #     This function perform correction 3 mentioned in https://arxiv.org/pdf/1704.06104.pdf  (Appendix)
    #     Any link that is outside of the actuall text, e.g. when predicted link > max_idx, is set to predicted_link== max_idx
    #     """

    #     max_segs = df.groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()
    #     df.loc[:, "max_seg"] = np.repeat(max_segs, df.groupby(level=0, sort=False).size().to_numpy())

    #     above = df["link"] >= df["max_seg"]
    #     below = df["link"] < 0

    #     df.loc[above | below, "link"] = df.loc[above | below, "max_seg"] - 1
    #     df.pop("max_seg")


    # def __ensure_homogeneous(self, task:str, df:pd.DataFrame):

    #     """
    #     ensures that the labels inside a segments are the same. For each segment we take the majority label 
    #     and use it for the whole span.
    #     """
    #     count_df = df.loc[:, ["seg_id", task]].value_counts(sort=False).to_frame()
    #     count_df.reset_index(inplace=True)
    #     count_df.rename(columns={0:"counts"}, inplace=True)
    #     count_df.drop_duplicates(subset=['seg_id'], inplace=True)

    #     print(count_df)
    #     print(count_df[task].to_numpy())
    #     print(lol)

    #     seg_lengths = df.groupby("seg_id", sort=False).size()
    #     most_common = np.repeat(count_df[task].to_numpy(), seg_lengths)

    #     df.loc[~df["seg_id"].isna(), task] = most_common


    # def decouple(self, task:str, subtasks:list, df: pd.DataFrame, level: str):
    #     subtask_preds = df[task].apply(lambda x: np.array(self._decouple_mapping[task][x]))


    # def _label_bios(self, df: pd.DataFrame):
    #     df["seg"] = "O"
    #     segs = df.groupby("seg_id", sort=False)
    #     for seg_id, seg_df in segs:
    #         df.loc[seg_df.index, "seg"] = ["B"] +  (["I"] * (seg_df.shape[0]-1))
    #     return df


    # def _fuse_subtasks(self, df: pd.DataFrame):

    #     for task in self.tasks:
    #         subtasks = task.split("+")
            
    #         if len(subtasks) <= 1:
    #             continue

    #         subtask_labels  = df[subtasks].apply(lambda row: '_'.join([str(x) for x in row]), axis=1)
    #         df[task] = subtask_labels
        
    #     return df




    # def _label_ams(self, df: pd.DataFrame, mode="pre"):

    #     if mode == "pre":
    #         df = self.__ams_as_pre(df)

    #     elif mode == "from_list":
    #         df = self.__ams_from_list(df)

    #     elif mode == "list_pre":
    #         raise NotImplementedError()

    #     return df


    # def __ams_as_pre(self, df: pd.DataFrame):
    #     df["am_id"] = np.nan
    #     df["adu_id"] = df["seg_id"].to_numpy()
    #     groups = df.groupby("sentence_id", sort=False)

    #     for sent_id, sent_df in groups:
            
    #         acs = sent_df.groupby("seg_id", sort=False)
    #         prev_ac_end = 0
    #         for ac_id, ac_df in acs:
                
    #             ac_start = min(ac_df["char_start"])
    #             ac_end = max(ac_df["char_end"])

    #             # more than previous end of ac and less than ac start
    #             cond1 = sent_df["char_start"] >= prev_ac_end 
    #             cond2 = sent_df["char_start"] < ac_start
    #             idxs = sent_df[cond1 & cond2].index

    #             #set the id to ac_id
    #             df.loc[idxs,["am_id", "adu_id"]] = ac_id

    #             # text = " ".join(df.loc[idxs, "text"].to_numpy())
    #             # if text:
    #             #     with open("/tmp/pe_ams.txt","a") as f:
    #             #         f.write(text+"\n")

    #             prev_ac_end = ac_end

    #     return df


    # def __ams_from_list(self, df: pd.DataFrame):

    #     df["am_id"] = np.nan
    #     groups = df.groupby("sentence_id", sort=False)

    #     for sent_id, sent_df in groups:
            
    #         acs = sent_df.groupby("seg_id", sort=False)

    #         for ac_id, ac_df in acs:

    #             tokens = ac_df["text"].to_list()
    #             am, am_indexes = find_am(tokens)

    #             if not am:
    #                 continue

    #             idx = ac_df.iloc[am_indexes].index
    #             df.loc[idx, "am_id"] = ac_id
    #             df.loc[idx, "ac_id"] = None

    #     return df







# def _overlap(j:int, target_seg_df: pd.DataFrame, pdf: pd.DataFrame) -> np.ndarray:

#     # target segment id
#     #j = target_seg_df["seg_id"].to_list()[0]

#     # predicted segments in the sample target df
#     pred_seg_ids = target_seg_df["PRED-seg_id"].dropna().to_list()

#     if not pred_seg_ids:
#         return np.array([None, None, None])
    
#     #best pred segment id
#     i = Counter(pred_seg_ids).most_common(1)[0][0]

#     #get token indexes
#     ps_token_ids = set(pdf.loc[[i], "token_id"]) #slowest part
#     ts_token_ids = set(target_seg_df["token_id"])

#     #calculate the overlap
#     overlap_ratio = len(ts_token_ids.intersection(ps_token_ids)) / max(len(ps_token_ids), len(ts_token_ids))

#     return np.array([i, j, overlap_ratio])


# def find_overlap(pred_df : pd.DataFrame, target_df : pd.DataFrame) -> Tuple[dict]:

#     # Create a new dfs which contain the information we need
#     pdf = pd.DataFrame({
#                         "token_id": pred_df["id"].to_numpy()
#                         }, 
#                         index = pred_df["seg_id"].to_numpy(),
#                         )
#     pdf = pdf[~pdf.index.isna()]

#     tdf = pd.DataFrame({
#                         "token_id": target_df["id"].to_numpy(),
#                         "seg_id": target_df["seg_id"].to_numpy(),
#                         "PRED-seg_id": pred_df["seg_id"].to_numpy(),
#                         }, 
#                         index = target_df["seg_id"].to_numpy(),
#                         )
    
#     tdf = tdf[~tdf.index.isna()]
    
#     if tdf.shape[0] == 0:
#         return {}, {}, {}, {}

#     # extract information about overlaps
#     overlap_info = np.vstack([ _overlap(j, tsdf, pdf) for j, tsdf in tdf.groupby("seg_id", sort=False)])    
  

#     # we then filter out all Nones, and filter out all cases where j match with more than one i.
#     # i.e. for each j we only selec the best i 
#     df = pd.DataFrame(overlap_info, columns = ["i", "j", "ratio"])
#     df = df.dropna()
#     df = df.sort_values("ratio")
#     top_matches = df.groupby("j", sort = False).first()

#     i  = top_matches["i"].to_numpy(int)
#     j = top_matches.index.to_numpy(int)
#     ratio = top_matches["ratio"].to_numpy(float)

#     i2ratio = dict(zip(i, ratio))
#     j2ratio = dict(zip(j, ratio))
#     i2j = dict(zip(i, j))
#     j2i = dict(zip(j, i))

#     return i2ratio, j2ratio, i2j, j2i














# def SOME
      
#         # if we are using TARGET segmentation results we  overwrite the 
#         # columns of seg_id with TARGET seg_id as well as TARGET labels for each
#         # task done in segmenation
#         if "seg" in key and self.use_target_segs:

#             self._pred_df["seg_id"] = self._df["seg_id"].to_numpy()

#             for subtask in key.split("+"):
#                 self._pred_df[subtask] = self._df[subtask].to_numpy()

#             #self.__add_overlap_info()
#             return
            
        
#         if level == "token":
#             mask = ensure_numpy(self.get("token", "mask")).astype(bool)
#             self._pred_df.loc[:, key] = ensure_numpy(value)[mask]


#         elif level == "seg":
#             mask = ensure_numpy(self.get("seg", "mask", pred = True)).astype(bool)
#             seg_preds = ensure_numpy(value)[mask]

#             # get the length of tokens for each seg 
#             tok_lens = self._pred_df.groupby("seg_id", sort=False).size().to_numpy()
            
#             # we spread the predictions on segments over tokens in TARGET segments
#             cond = ~self._pred_df["seg_id"].isna()

#             # expand the segment prediction for all their tokens 
#             token_preds = np.repeat(seg_preds, tok_lens)
            
#             #set the predictions for all rows which belong to a TARGET segment
#             self._pred_df.loc[cond, key] = token_preds


#         elif level == "p_seg":

#             #get the lengths of each segment
#             seg_lengths = self._pred_df.groupby("seg_id", sort=False).size().to_numpy()
            
#             #expand the predictions over the tokens in the segments
#             token_preds = np.repeat(value, seg_lengths)

#             # as predicts are given in seg ids ordered from 0 to nr predicted segments
#             # we can just remove all rows which doesnt belong to a predicted segments and 
#             # it will match all the token preds and be in the correct order.
#             self._pred_df.loc[~self._pred_df["seg_id"].isna(), key] = token_preds


#         self._pred_df = self.label_encoder.validate(
#                                                     task = key,
#                                                     df = self._pred_df,
#                                                     level = level,
#                                                     )
        

#         # creating target_ids for links
#         if key == "link" or "link" in key.split("+"):


#             # remove rows outside segments
#             is_not_nan = ~self._pred_df["seg_id"].isna()

            
#             for si, sample_df in self._pred_df.groupby("sample_id", sort = False):
                
#                 # remove samples that doens have segments if we are predicting on segments
#                 segs = sample_df.groupby("seg_id", sort = False)
            
#                 # it might be helpfult to keep track on the global seg_id of the target
#                 # i.e. the seg_id of the linked segment
#                 seg_first = segs.first()

#                 links = seg_first["link"].to_numpy(dtype=int)

#                 target_ids = seg_first.index.to_numpy()[links]

#                 #print(np.repeat(target_ids, segs.size().to_numpy()))

#                 is_sample_row = self._pred_df["sample_id"] == si
#                 row_mask = is_not_nan & is_sample_row

#                 # exapnd target_id over the rows
#                 self._pred_df.loc[row_mask, "target_id"] = np.repeat(target_ids, segs.size().to_numpy())
        


# def get_overlapping_targets(self, level:str, key:str, threshold:float = 0.5):

#         preds = self.get(level, key, pred =True)
#         targets = self.get(level, key)

#         # we also have information about whether the seg_id is a true segments 
#         # and if so, which TRUE segmentent id it overlaps with, and how much
#         i2ratio, j2ratio, i2j, j2i = find_overlap(
#                                                 target_df = self._df,  
#                                                 pred_df = self._pred_df
#                                                 )


#         # we figure out which segments are overlaping over 50%  with a target segment
#         # each target segment will then only match with at most one predicted segment
#         overlap_mask = torch.tensor([i2ratio.get(i, 0) > threshold for i, _ in self._pred_df.groupby("seg_id", sort=False)])


#         # we havent included paddings to we will split,pad and the flatten to include padding
#         plens = ensure_list(self.get("seg", "lengths", pred = True))
#         overlap_mask = pad_sequence(torch.split(overlap_mask, plens)).view(-1)


#         # # we also need to figure out which target we are ignoring
#         target_overlap_mask = torch.tensor([j2ratio.get(j,0) > threshold for j, _ in self._df.groupby("seg_id", sort=False)])

#         # we havent included paddings to we will split,pad and the flatten to include padding
#         tlens = ensure_list(self.get("seg","lengths"))
#         target_overlap_mask = pad_sequence(torch.split(target_overlap_mask, tlens)).view(-1)
        

#         # then we create a new target tensor which match the shape of the preds
#         # here the default label will be 0 for all task except link where the 
#         # label will be the index of of the segment, i.e. pointing to itself
#         if key == "link":
#             new_targets = torch.arange(preds.size(-1)).repeat(preds.size(0)).type(torch.long)

#             overlap_mask
#             overlap_mask = torch.tensor([i2ratio.get(it, 0) > threshold for it, _ in self._pred_df.groupby("target_id", sort=False)])


#             targets.view(-1)[target_overlap_mask]


#             seg_ids = self._df.groupby("seg_id", sort=False).groups.keys()
#             target_ids = self._df.groupby("seg_id", sort=False).groups.keys()


#             target_ids[overlap_mask]



#             targets.view(-1)[target_overlap_mask]




#             new_targets[overlap_mask] = targets.view(-1)[target_overlap_mask]

#         else:
#             new_targets = torch.zeros(preds.shape, dtype=torch.long).view(-1)

#             # then we set the target labels for all spaces where there is an overlap to the target label
#             new_targets[overlap_mask] = targets.view(-1)[target_overlap_mask]


#         #lastly we set the padding target values to -1, so we can ignore them in the loss function
#         new_targets[~self.get("seg","mask", pred = True).view(-1)] = -1

#         new_targets = new_targets.view(preds.shape)

#         return new_targets








    
# def __get_column_values(self, df: pd.DataFrame, level: str, key:str):

#         if level == "token":
#             flat_values = df.loc[:, key].to_numpy()
#         else:
#             flat_values = df.groupby(f"{level}_id", sort = False).first().loc[:, key].to_numpy()

#         if isinstance(flat_values[0], str):
#             return flat_values
#         else:
#             return torch.LongTensor(flat_values)


# def __get_span_idxs(self, df: pd.DataFrame, level:str ):

#         if level == "am":
#             ADU_start = df.groupby("adu_id", sort=False).first()["sample_token_id"].to_numpy()
#             ADU_end = df.groupby("adu_id", sort=False).last()["sample_token_id"].to_numpy() + 1 

#             AC_lens = df.groupby("seg_id", sort=False).size().to_numpy()

#             AM_start = ADU_start
#             AM_end = ADU_end - AC_lens

#             return torch.LongTensor(np.column_stack((AM_start, AM_end)))

#         else:

#             start_tok_ids = df.groupby(f"{level}_id", sort=False).first()["sample_token_id"].to_numpy()
#             end_tok_ids = df.groupby(f"{level}_id", sort=False).last()["sample_token_id"].to_numpy() + 1

#             return torch.LongTensor(np.column_stack((start_tok_ids, end_tok_ids)))


# def __get_mask(self, level:str, pred : bool = False):
#         return create_mask(self.get(level, "lengths", pred = pred), as_bool = True)


# def __get_lengths(self, df: pd.DataFrame, level:str):

#         if level == "token":
#             return torch.LongTensor(df.groupby(level=0, sort = False).size().to_numpy())
#         else:
#             return torch.LongTensor(df.groupby(level=0, sort=False)[f"{level}_id"].nunique().to_numpy())


# def __get_pretrained_embeddings(self, df:pd.DataFrame, level:str, flat:bool):

#         if level == "token":
#             embs = self._pretrained_features["word_embs"]
#         else:
#             embs = self._pretrained_features["seg_embs"]

#         embs = embs[:, :max(self.__get_lengths(df, level)), :]

#         if flat:
#             embs = embs[self.__get_mask("level")]

#         return torch.tensor(embs, dtype = torch.float)


# def __add_link_matching_info(self, pair_df:pd.DataFrame, j2i:dict):


#         def check_true_pair(row, mapping):

#             p1 = row["p1"]
#             p2 = row["p2"]
#             dir = row["direction"]
            
#             source = p2 if dir == 2 else p1
#             target = p1 if dir == 2 else p2

#             if source not in mapping:
#                 return False
#             else:
#                 correct_target = mapping[source]
#                 return correct_target == target

        
#         j_jt = self._df.loc[:, ["seg_id", "target_id"]].dropna()

#         # maps a true source to the correct target using the ids of predicted pairs
#         source2target = {
#                         j2i.get(j, "NONE"): j2i.get(jt, "NONE")
#                         for j,jt in zip(j_jt["seg_id"], j_jt["target_id"])
#                         }

#         if "NONE" in source2target:
#             source2target.pop("NONE")


#         if not source2target:
#             pair_df["true_link"] = False
#             return
        
#         pair_df["true_link"] = pair_df.apply(check_true_pair, axis = 1, args = (source2target, ))


# def __create_pair_df(self, df: pd.DataFrame, pred :bool):


#         def set_id_fn():
#             pair_dict = dict()

#             def set_id(row):
#                 p = tuple(sorted((row["p1"], row["p2"])))

#                 if p not in pair_dict:
#                     pair_dict[p] = len(pair_dict)

#                 return pair_dict[p]

#             return set_id


#         # we also have information about whether the seg_id is a true segments 
#         # and if so, which TRUE segmentent id it overlaps with, and how much
#         i2ratio, j2ratio, i2j, j2i = find_overlap(
#                                                 target_df = self._df,  
#                                                 pred_df = self._pred_df
#                                                 )



#         first_df = df.groupby("seg_id", sort=False).first()
#         first_df.reset_index(inplace=True)

#         last_df = df.groupby("seg_id", sort=False).last()
#         last_df.reset_index(inplace=True)


#         if pred:
#             first_target_df = self._df.groupby("seg_id", sort=False).first()
#             j2link_label = {j:row["link_label"] for j, row in first_target_df.iterrows()}
#             link_labels = [-1 if i not in i2j else j2link_label.get(i2j[i], -1) for i in first_df.index.to_numpy()]
#             first_df["link_label"] = link_labels


#         # we create ids for each memeber of the pairs
#         # the segments in the batch will have unique ids starting from 0 to 
#         # the total mumber of segments
#         p1, p2 = [], []
#         j = 0
#         for _, gdf in df.groupby("sample_id", sort = False):
#             n = len(gdf.loc[:, "seg_id"].dropna().unique())
#             sample_seg_ids = np.arange(
#                                         start= j,
#                                         stop = j+n
#                                         )
#             p1.extend(np.repeat(sample_seg_ids, n).astype(int))
#             p2.extend(np.tile(sample_seg_ids, n))
#             j += n
    
#         # setup pairs
#         pair_df = pd.DataFrame({
#                                 "p1": p1,
#                                 "p2": p2,
#                                 })
        


#         if not len(pair_df.index):
#             return pd.DataFrame()

#         # create ids for each NON-directional pair
#         pair_df["id"] = pair_df.apply(set_id_fn(), axis=1)

#         #set the sample id for each pair
#         pair_df["sample_id"] = first_df.loc[pair_df["p1"], "sample_id"].to_numpy()


#         #set true the link_label
#         #pair_df["link_label"] = first_df.loc[pair_df["p1"], "link_label"].to_numpy()

#         #set start and end token indexes for p1 and p2
#         pair_df["p1_start"] = first_df.loc[pair_df["p1"], "sample_token_id"].to_numpy()
#         pair_df["p1_end"] = last_df.loc[pair_df["p1"], "sample_token_id"].to_numpy()

#         pair_df["p2_start"] = first_df.loc[pair_df["p2"], "sample_token_id"].to_numpy()
#         pair_df["p2_end"] = last_df.loc[pair_df["p2"], "sample_token_id"].to_numpy()

#         # set directions
#         pair_df["direction"] = 0  #self
#         pair_df.loc[pair_df["p1"] < pair_df["p2"], "direction"] = 1 # ->
#         pair_df.loc[pair_df["p1"] > pair_df["p2"], "direction"] = 2 # <-


#         # mask for where p1 is a source        
#         p1_source_mask = np.logical_or(pair_df["direction"] == 0 , pair_df["direction"] == 1)
#         pair_df.loc[p1_source_mask, "link_label"] = first_df.loc[pair_df.loc[p1_source_mask, "p1"], "link_label"].to_numpy()

#         #where p2 is a source
#         p2_source_mask = pair_df["direction"] == 2
#         pair_df.loc[p2_source_mask, "link_label"] = first_df.loc[pair_df.loc[p2_source_mask, "p2"], "link_label"].to_numpy()



#         self.__add_link_matching_info(pair_df, j2i)


#         if pred:
#             pair_df["p1-ratio"] = pair_df["p1"].map(i2ratio)
#             pair_df["p2-ratio"] = pair_df["p2"].map(i2ratio)
#         else:
#             pair_df["p1-ratio"] = 1
#             pair_df["p2-ratio"] = 1


#         return pair_df


# def __get_df_data(self,
#                     level : str, 
#                     key : str, 
#                     flat : bool = False, 
#                     pred : bool = False,
#                     ) -> Union[Tensor, list, np.ndarray]:


#         df = self._pred_df if pred else self._df

    
#         if key == "lengths":
#             data =  self.__get_lengths(df, level)

#         # elif key == "lengths_tok":
#         #     data = self.__seg_tok_lengths(df, level)

#         elif key == "embs":
#             data =  self.__get_pretrained_embeddings(df, level, flat = flat)

#         elif key == "mask":
#             data = self.__get_mask(level, pred = pred)

#         else:
#             if key == "span_idxs":
#                 data = self.__get_span_idxs(df, level)
#             else:
#                 data = self.__get_column_values(df, level, key)

#             if len(data) == 0:
#                 return data

#             if isinstance(data[0], str):
#                 return data

#             if not flat:

#                 if level == "am" and key == "span_idxs":
#                     level = "adu"

#                 lengths = ensure_list(self.get(level, "lengths", pred = pred))

#                 data =  pad_sequence(
#                                     torch.split(
#                                                 data, 
#                                                 lengths
#                                                 ), 
#                                     batch_first = True,
#                                     padding_value = -1 if self._task_regexp.search(key) else 0,
#                                     )
        
#         return data


# def __get_pair_df_data(self,
#                     key : str, 
#                     bidir : bool = True,   
#                     ) -> Union[Tensor, list, np.ndarray]:


#         if not hasattr(self, "_pair_df"):

#             pred = not self.use_target_segs

#             self._pair_df = self.__create_pair_df(
#                                                 df = self._pred_df if pred else self._df,
#                                                 pred = pred
#                                                 )

#         pair_df = self._pair_df
        
#         if not len(self._pair_df.index):
#             return []

#         if not bidir:
#             pair_df = pair_df[pair_df["direction"].isin([0,1]).to_numpy()]

#         if key == "lengths":
#             sample_ids = list(self._df.groupby("sample_id", sort = False).groups.keys())
#             sample_pair_lens = pair_df.groupby("sample_id", sort = False).size().to_dict()
#             data = [sample_pair_lens.get(i, 0) for i in sample_ids]

#         else:
#             data = torch.LongTensor(pair_df[key].to_numpy())

#         return data

    

# #basics
# import re
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Sequence
# from tqdm import tqdm


# #segnlp
# from segnlp import utils
# from segnlp import get_logger
# from segnlp.resources.am import find_am

# logger = get_logger("LABELER")

# class Labeler:


#     def _encode_labels(self, df):

#         for task in self.all_tasks:
#             self.label_encoder.encode( 
#                                         task = task,
#                                         df = df
#                                         )
#         return df


#     def _decode_labels(self, df: pd.DataFrame):
#         raise NotImplementedError


#     def _label_spans(self, df:pd.DataFrame, span_labels:dict):

#         def label_f(row, span_labels):
#             return span_labels.get(int(row["char_end"]),{})

#         df = pd.concat([df,df.apply(label_f, axis=1, result_type="expand", args=(span_labels,))], axis=1)
#         df = df.astype({'seg_id': 'float64'})
#         return df


#     def _fuse_subtasks(self, df: pd.DataFrame):

#         for task in self.tasks:
#             subtasks = task.split("+")
            
#             if len(subtasks) <= 1:
#                 continue

#             subtask_labels  = df[subtasks].apply(lambda row: '_'.join([str(x) for x in row]), axis=1)
#             df[task] = subtask_labels
        
#         return df


#     def _label_bios(self, df: pd.DataFrame):
#         df["seg"] = "O"
#         segs = df.groupby("seg_id", sort=False)
#         for seg_id, seg_df in segs:
#             df.loc[seg_df.index, "seg"] = ["B"] +  (["I"] * (seg_df.shape[0]-1))
#         return df









# class DatasetProcessor:


#     def _init_dataset_processor(self):

#         # argumentative markers
#         self.argumentative_markers : bool = False 
#         if "am" in self.other_levels:
#             self.argumentative_markers = True

#             if self.dataset_name == "MTC":
#                 self.am_extraction = "from_list"
#             else:
#                 self.am_extraction = "pre"

#         # preprocessing
#         self._need_bio : bool = "seg" in self.subtasks
#         self._labeling : bool = True
#         self._removed : int = 0


#     def __process_dataset(self, dataset : DataSet) -> pd.DataFrame:

#         self._n_samples = 0

#         sample_dfs = []

#         for i in tqdm(range(len(dataset)), desc=f"Prerocessing Dataset (nlp = {self._nlp_name})"):
#             doc = dataset[i]

#             span_labels = doc.get("span_labels", None)
#             doc = doc["text"]   

#             doc_df = self._process_text(doc)

#             if self.input_level != self.sample_level:
#                 samples = doc_df.groupby(f"{self.sample_level}_id", sort=False)
#             else:
#                 samples = [(None, doc_df)]

#             for _, sample in samples:

#                 tok_sample_id = np.full(sample.index.shape, fill_value = self._n_samples)
#                 sample["sample_id"] = tok_sample_id
#                 sample["sample_token_id"] = sample[f"{self.sample_level}_token_id"].to_numpy()

#                 if span_labels:
#                     sample = self._label_spans(sample, span_labels)
                
#                 if self._need_bio:
#                     sample = self._label_bios(sample)
                
#                 sample = self._fuse_subtasks(sample)
#                 sample = self._encode_labels(sample)

#                 if self.argumentative_markers:
#                     sample = self._label_ams(sample, mode=self.am_extraction)
                

#                 if self.sample_level != "sentence":

#                     # remove samples that doens have segments if we are predicting on segments
#                     segs = sample.groupby("seg_id", sort = False)
#                     seg_length = len(segs)
#                     if self.prediction_level == "seg" and seg_length == 0:
#                         continue

#                     # it might be helpfult to keep track on the global seg_id of the target
#                     # i.e. the seg_id of the linked segment
#                     seg_first = segs.first()
#                     target_ids = seg_first.index.to_numpy()[seg_first["link"].to_numpy()]
#                     sample.loc[~sample["seg_id"].isna() ,"target_id"] = np.repeat(target_ids, segs.size().to_numpy())
                

#                 sample.index = tok_sample_id
#                 sample_dfs.append(sample)
#                 self._n_samples += 1

#         #concatenate all samples to a big dataframe
#         df = pd.concat(sample_dfs)

#         #save csv 
#         df.to_csv(self._path_to_df)

#         return df





   
# # basics
# import pandas as pd
# import numpy as np
# from tqdm.auto import tqdm

# # h5py
# import h5py

# #segnlp
# from segnlp import get_logger


# logger = get_logger(__name__)
 
# class PretrainedFeatureExtractor:


#     def _init_pretrained_feature_extractor(self, pretrained_features):

#         # pretrained featues
#         self.feature2model : dict = {fm.name:fm for fm in pretrained_features}
#         self.features : list = list(self.feature2model.keys())
#         self._feature_groups : set = set([fm.group for fm in pretrained_features])
#         self.feature2dim : dict = {fm.name:fm.feature_dim for fm in pretrained_features}
#         self.feature2dim.update({
#                                 group:sum([fm.feature_dim for fm in pretrained_features if fm.group == group]) 
#                                 for group in self._feature_groups
#                                 })

#         self.feature2param = {f.name:f.params for f in pretrained_features}
#         self._use_pwf : bool = "word_embs" in self._feature_groups
#         self._use_psf : bool = "seg_embs" in self._feature_groups


#     def __extract_sample_features(self, sample:pd.DataFrame):
            
#         feature_dict = {}
#         sample_length = sample.shape[0]

#         for feature, fm in self.feature2model.items():
    
#             if fm.level == "doc" and self.prediction_level == "seg":
                
#                 segs = sample.groupby("seg_id", sort = False)
#                 feature_matrix = np.zeros((len(segs), fm.feature_dim))
#                 for i,(seg_id, seg_df) in enumerate(segs):
#                     # sent.index = sent["id"]
#                     data = sample[sample["seg_id"] == seg_id]

#                     if self.argumentative_markers:
#                         am = sample[sample["am_id"] == seg_id]
#                         data = pd.concat((am,data))

#                     #adu.index = adu.pop("seg_id")
#                     feature_matrix[i] = fm.extract(data)


#             elif fm.level == "word":
#                 # context is for embeddings such as Bert and Flair where the word embeddings are dependent on the surrounding words
#                 # so for these types we need to extract the embeddings per context. E.g. if we have a document and want Flair embeddings
#                 # we first divide the document up in sentences, extract the embeddigns and the put them bsegk into the 
#                 # ducument shape.
#                 # Have chosen to not extract flair embeddings with context larger than "sentence".
#                 if fm.context and fm.context != self.sample_level:

#                     contexts = sample.groupby(f"{fm.context}_id", sort = False)

#                     sample_embs = []
#                     for _, context_data in contexts:
#                         sample_embs.extend(fm.extract(context_data)[:context_data.shape[0]])

#                     feature_matrix = np.array(sample_embs)
            
#                 else:
#                     #feature_matrix[:sample_length] = fm.extract(sample)[:sample_length]
#                     feature_matrix = fm.extract(sample)[:sample_length]

#             else:
#                 feature_matrix = fm.extract(sample)[:sample_length]


#             if fm.group not in feature_dict:
#                 feature_dict[fm.group] = {
#                                         "level": "seg" if fm.level == "doc" else "token",
#                                         "data":[]
#                                         }
            

#             feature_dict[fm.group]["data"].append(feature_matrix)


#         outputs = {}
#         for group_name, group_dict in feature_dict.items():

#             if len(group_dict["data"]) > 1:
#                 outputs[group_name] =  np.concatenate(group_dict["data"], axis=-1)
#             else:
#                 outputs[group_name] = group_dict["data"][0]

#         return outputs


#     def _preprocess_pretrained_features(self, df : pd.DataFrame) -> None:

#         if self._use_pwf:
#             #logger.info("Creating h5py file for pretrained word features ... ")

#             max_toks = max(df.groupby(level = 0, sort=False).size())
#             fdim = self.feature2dim["word_embs"]

#             h5py_pwf = h5py.File(self._path_to_pwf, "w")
#             h5py_pwf.create_dataset(
#                                     "word_embs", 
#                                     data = np.random.random((self._n_samples, max_toks, fdim)), 
#                                     dtype = np.float64, 
#                                     )
    


#         if self._use_psf:
#             #logger.info("Creating h5py file for pretrained segment features ... ")

#             max_segs = max(df.groupby(level = 0, sort=False)["seg_id"].nunique())
#             fdim = self.feature2dim["seg_embs"]

#             h5py_psf = h5py.File(self._path_to_psf, "w")
#             h5py_psf.create_dataset(
#                                     "seg_embs", 
#                                     data = np.random.random((self._n_samples, max_segs, fdim)), 
#                                     dtype = np.float64, 
#                                     )
    

#         for i, sample in tqdm(df.groupby(level = 0), desc="Preprocessing Pretrained Features"):
#             feature_dict = self.__extract_sample_features(sample)

#             if self._use_pwf:
#                 t, _ = feature_dict["word_embs"].shape
#                 h5py_pwf["word_embs"][i, :t, :] = feature_dict["word_embs"]
            
#             if self._use_psf:
#                 s, _ = feature_dict["seg_embs"].shape
#                 h5py_psf["seg_embs"][i, :s, :] = feature_dict["seg_embs"]

