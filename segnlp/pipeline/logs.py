
#basics
import os
from glob import glob
from random import random
import pandas as pd
import re


# segnlp
from segnlp import utils

class Logs:

    def _init_logs(self) -> None:
        self._max_lines : int = 10000
        self._current_lines : int = 0
        self._current_random_seed = None
        self._current_split = None


    # def _remove_logs(self, hp_id, random_seed:int = None):
    #     pass


    def _load_logs(self) -> pd.DataFrame:
    
        log_dfs = {}
        for hp_id in self.hp_ids:

            # log folder
            log_folder_path = os.path.join(self._path_to_logs, hp_id)

            # log files 
            split_dfs = {}
            for split in ["train", "val", "test"]:
                log_files = glob(log_folder_path + f"/{split}/*")


                if not log_files:
                    continue

                # log dataframes
                split_dfs[split] = pd.concat([pd.read_csv(fp) for fp in log_files])

            log_dfs[hp_id] = split_dfs

        return log_dfs


    def __check_create_log_folder(self, hp_id:str, split:str):
        folder_path = os.path.join(self._path_to_logs, hp_id, split)
        os.makedirs(folder_path, exist_ok = True)
        return folder_path


    def __set_create_log_file(self, folder_path :str,  file_name:str):

        #create log file
        self.log_file = os.path.join(folder_path, f"{file_name}.log")

        # create the file
        open(self.log_file, "w")
        
    
    def _log_epoch(self, 
                epoch : int,
                split : str,
                hp_id : str,
                random_seed : int,
                epoch_metrics : dict,
                cv : int,
                use_target_segs :bool = False,
                freeze_segment_module : bool = False,
                ):                            

        # create dict
        log_dict = {
                    "epoch": epoch,
                    "split": split,
                    "hp_id":hp_id,
                    "random_seed": random_seed,
                    "cv":cv,
                    "use_target_segs" : use_target_segs,
                    "freeze_segment_module": freeze_segment_module
                    }
        log_dict.update(epoch_metrics)

        log_folder_path = self.__check_create_log_folder(hp_id, split)

        new_file = False
        if self._current_random_seed != random_seed or self._current_split != split:
            self.__set_create_log_file(
                                        folder_path = log_folder_path, 
                                        file_name = random_seed
                                        )
            new_file = True
            self._current_random_seed = random_seed


        with open(self.log_file, "a") as f:
            
            # create a header if file is new
            if new_file:
                f.write(",".join(list(log_dict.keys())) + "\n")
            
            # row
            f.write(",".join([str(v) for v in log_dict.values()]) + "\n")

        self._current_lines += 1
