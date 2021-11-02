
#basics
import shutil
from typing import List, Tuple, Dict, Callable, Union
import json
import numpy as np
import os
import pwd
import pandas as pd
from glob import glob


#pytorch
import torch

# segnlp
from .model_eval import ModelEval
from .baselines import Baseline
from .rank import Rank

from segnlp.datasets.base import DataSet
import segnlp.utils as utils
from segnlp.seg_model import SegModel


user_dir = pwd.getpwuid(os.getuid()).pw_dir


class Experiment(
                ModelEval,
                Baseline,
                Rank
                ):
    
    def __init__(self,
                id:str,
                dataset: DataSet,
                metric:str,
                evaluation_method:str = "default",
                n_random_seeds: int = 6,
                root_dir:str =f"{user_dir}/.segnlp/", #".segnlp/pipelines"  
                overwrite: bool = False,
                ):

        #general
        self.id : str = id
        self.evaluation_method : str = evaluation_method
        self.metric : str = metric
        self.training : bool = True
        self.testing : bool = True
        self.root_dir = root_dir
        self.dataset = dataset
        self.dataset_name = dataset.name()
        self.n_random_seeds = n_random_seeds

        # task info
        self.dataset_level : str = dataset.level
        self.prediction_level : str = dataset.prediction_level
        self.sample_level : str = dataset.sample_level
        self.input_level : str = dataset.level
        self.tasks : list = dataset.tasks
        self.subtasks : list = dataset.subtasks
        self.task_labels : Dict[str,list] = dataset.task_labels
        self.all_tasks : list = sorted(set(self.tasks + self.subtasks))

        # init files
        self.__init__dirs_and_files(overwrite = overwrite)

        # create config
        self.config = self.__create_dump_config()


    def __init__dirs_and_files(self, overwrite : bool):

        self._path = os.path.join(self.root_dir, self.id)

        if overwrite:

            new_loc = os.path.join("/tmp/", self.id)
            if os.path.exists(new_loc):
                shutil.rmtree(new_loc)

            shutil.move(self._path, new_loc)

        os.makedirs(self._path, exist_ok=True)

        # Setup the main folder names
        self._path_to_models  : str = os.path.join(self._path, "models")
        self._path_to_logs : str = os.path.join(self._path, "logs")
        self._path_to_hps : str = os.path.join(self._path, "hps")


        # create the main folders
        os.makedirs(self._path_to_models, exist_ok=True)
        os.makedirs(self._path_to_logs, exist_ok=True)
        os.makedirs(self._path_to_hps, exist_ok=True)

        # set up files paths
        self._path_to_bs_scores : str = os.path.join(self._path, "baseline_scores.json") # for hp history
        self._path_to_rankings : str = os.path.join(self._path, "rankings.csv") # for hp history


    @property
    def hp_ids(self):
        hp_ids = [fp.replace(".json","") for fp in os.listdir(self._path_to_hps)]
        return hp_ids


    @property
    def hp_configs(self):
        return [utils.load_json(fp) for fp in glob(self._path_to_hps + "/*.json")]


    def __check_config(self, config:dict):

        # create a key for the id
        config_key = utils.create_uid(str(config))

        key_file = os.path.join(self._path, "key.txt")

        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                key = key_file.read().strip()

            if key != config_key:
                raise RuntimeError(f"Current config is not the same as the config found in {self._path}. Either change the id of the pipeline or make sure all the paramaters the same as for {self.id}")
  

    def __create_dump_config(self) -> dict:

        #create and save config
        config = dict(
                            id = self.id,
                            dataset = self.dataset_name,
                            evaluation_method = self.evaluation_method,
                            tasks = self.tasks,
                            subtasks =  self.subtasks,
                            all_tasks = self.all_tasks,
                            task_labels = self.task_labels,
                            prediction_level = self.prediction_level,
                            sample_level = self.sample_level,
                            input_level = self.input_level,
                            )
        
        self.__check_config(config)

        config_fp = os.path.join(self._path, "config.json")
        if not os.path.exists(config_fp):
            with open(config_fp, "w") as f:
                json.dump(config, f, indent=4)

        return config


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