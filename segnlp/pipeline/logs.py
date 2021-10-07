
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


    def _remove_logs(self, hp_id, random_seed:int = None):
        pass
    


    def _load_logs(self) -> pd.DataFrame:
    
        log_dfs = {}
        all_log_files = glob(self._path_to_logs + "/*")

        for hp_id in self.hp_ids:

            # get the hyperparamater config
            hp_info = utils.load_json(os.path.join(self._path_to_hps, hp_id + ".json"))

            # get the random seeds used
            random_seeds = hp_info["random_seeds"]

            log_files = []
            for random_seed in random_seeds:
                pattern = re.compile(f".*{self.hp_ids}-{random_seed}-.*.log")
                log_files.extend(filter(pattern.match, all_log_files))


            log_dfs[hp_id] = pd.concat([pd.read_csv(fp) for fp in log_files])

        return log_dfs


    def __check_create_log_folder(self, folder_name:str):
        folder_path = os.path.join(self._path_to_logs, folder_name)
        os.makedirs(os.path.join(self._path_to_logs, folder_name), exist_ok = True)
        return folder_path


    def __set_create_log_file(self, folder_path :str,  file_name:str):

        #set a file path
        self.log_file = os.path.join(folder_path, f"{file_name}-{len(os.listdir(folder_path))}.log")

        #create the file
        open(self.log_file, "w")



    def _log_epoch(self, 
                epoch : int,
                split : str,
                hp_id : str,
                random_seed : int,
                epoch_metrics : dict,
                cv : int,
                use_target_segs :bool,
                freeze_segment_module : bool,
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

        new_file = False
        if not hasattr(self, "log_file"):
            log_folder_path = self.__check_create_log_folder(hp_id)
            self.__set_create_log_file(
                                            folder_path = log_folder_path, 
                                            file_name = random_seed
                                            )
            new_file = True

        with open(self.log_file, "a") as f:
            
            # create a header if file is new
            if new_file:
                f.write(",".join(list(log_dict.keys())) + "\n")
            
            # row
            f.write(",".join([str(v) for v in log_dict.values()]) + "\n")

        self._current_lines += 1
