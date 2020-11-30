

#basics
import os
import numpy as np
from typing import Union, List, Dict, Tuple

#sklearn
from sklearn.model_selection import KFold

#hotam
from hotam import get_logger
from hotam.utils import one_tqdm



logger = get_logger("SPLIT-UTILS")

class SplitUtils:


    def __contain_ac(self, ID):
        # if there is a an AC_ID it should return an int
        return len(self.level_dfs["token"].loc[ID]["ac_id"].dropna().unique())

    @one_tqdm(desc="Changing Split Level")
    def _change_split_level(self):
        """changes the level on which the splits are on. If a dataset have premade splits for documents 
        and you want to predict on sentences, the split ids are remade to match the orignal splits for documents.

        Raises
        ------
        RuntimeError
            Given splits need to match level else changing level is not doable.
        """
        
        #we get the set of split ids and compare it to a level to find witch it matches
        split_sets = set()
        for split_id, splits in self.splits.items():
            split_sets.update(splits["train"].tolist())
            split_sets.update(splits["val"].tolist())
            split_sets.update(splits["test"].tolist())

        split_size = len(split_sets)

        if self.nr_samples == split_size:
            return

        #lets find the level given splits are of
        # TODO: make this more exhaustive by making sure all ids exist in the level df
        split_id_level = False
        for level, df in self.level_dfs.items():
            
            level_size = df.shape[0]
            if level_size == split_size:
                split_id_level = f"{level}_id"
                break
                
        if not split_id_level:
            raise RuntimeError("No matching level found for given splits. Given splits need to match level else changing level is not doable.")

        self.original_splits = self.splits
        for split_id, splits in self.splits.copy().items():

            for split_type, split in splits.copy().items(): 
                cond = self.level_dfs[self.sample_level][split_id_level].isin(split)
                new_split_ids = self.level_dfs[self.sample_level]["id"][cond].to_numpy()

                if self.prediction_level == "ac":
                    new_split_ids = np.array([i for i in new_split_ids if self.__contain_ac(i)])

                self.splits[split_id][split_type] = new_split_ids


    def _update_splits(self):
        """
        updates the splits based on the found duplicate ids 
        """

        #then we need to update the splits and remove duplicate ids
        for split_id, splits in self.splits.copy().items():
            #splits = self.splits[split_id]["splits"].copy()
            for split_type, split in splits.copy().items():
                self.splits[split_id][split_type] = split[~np.isin(split,self.duplicate_ids)]


    def __create_splits(self):
        raise NotImplementedError("No support for that splitting alternative yet")


    def add_splits(self, splits:List[list]):
        """adding splits

        Parameters
        ----------
        splits : List[list]
            lists of ids for each split
        """
        assert len(splits)
        assert isinstance(splits, dict)
        self.nr_splits = len(splits)
        self.splits = splits
