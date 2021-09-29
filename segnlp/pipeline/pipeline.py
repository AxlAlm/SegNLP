
#basics
import shutil
from typing import List, Tuple, Dict, Callable, Union
import json
import numpy as np
import os
import pwd



#pytorch
import torch

# segnlp
from .loop_train import TrainLoop
from .loop_test import TestLoop
from .loop_hp_tune import HPTuneLoop
from .dataset_processor import DatasetProcessor
from .labeler import Labeler
from .text_processor import TextProcessor
from .splitter import Splitter
from .pretrained_feature_extractor import PretrainedFeatureExtractor
from .logs import Logs

from segnlp import get_logger
from segnlp.datasets.base import DataSet
import segnlp.utils as utils
from segnlp import models
from segnlp.models.base import BaseModel

logger = get_logger("PIPELINE")
user_dir = pwd.getpwuid(os.getuid()).pw_dir


class Pipeline(
                TextProcessor, 
                DatasetProcessor,
                PretrainedFeatureExtractor,
                Labeler, 
                Splitter,
                HPTuneLoop,
                TrainLoop, 
                TestLoop,
                Logs,
                ):
    
    def __init__(self,
                id:str,
                dataset:Union[str, DataSet],
                model:Union[torch.nn.Module, str],
                metric:str,
                pretrained_features:list = [],
                other_levels:list = [],
                evaluation_method:str = "default",
                root_dir:str =f"{user_dir}/.segnlp/", #".segnlp/pipelines"  
                overwrite: bool = False,
                ):

        #general
        self.id : str = id
        self.model : BaseModel = getattr(models, model) if isinstance(model,str) else model
        self.model_name = str(self.model)
        self.evaluation_method : str = evaluation_method
        self.metric : str = metric
        self.training : bool = True
        self.testing : bool = True
        self.root_dir = root_dir
        self.other_levels = other_levels
        self.dataset_name = dataset.name()

        #setup pipeline  root folder
        self._path : str = os.path.join(self.root_dir, self.id)

        # task info
        self.dataset_level : str = dataset.level
        self.prediction_level : str = dataset.prediction_level
        self.sample_level : str = dataset.sample_level
        self.input_level : str = dataset.level
        self.tasks : list = dataset.tasks
        self.subtasks : list = dataset.subtasks
        self.task_labels : Dict[str,list] = dataset.task_labels
        self.all_tasks : list = sorted(set(self.tasks + self.subtasks))
        self.label_encoder : utils.LabelEncoder = utils.LabelEncoder(task_labels = self.task_labels)
   
        # init modules
        self._init_dataset_processor()
        self._init_pretrained_feature_extractor(pretrained_features)
        self._init_text_processor()
        self._init_logs()

        # create config
        self.config = self.__create_dump_config()

        # init files
        self.__init__dirs_and_files(overwrite = overwrite)

        #processed the data
        self.__preprocess_dataset(dataset)

        # create split indexes 
        self._set_splits(
                        premade_splits = dataset.splits,
                        )

        # after we have processed data we will deactivate preprocessing so we dont keep
        # large models only using in preprocessing in memory
        self.deactivate()


    def __init__dirs_and_files(self, overwrite : bool):

        if overwrite:
            #logger.info(f"Overriding all data in {self._path} by moving existing folder to /tmp/ and creating a new folder")

            new_loc = os.path.join("/tmp/", self.id)
            if os.path.exists(new_loc):
                shutil.rmtree(new_loc)

            shutil.move(self._path, new_loc)

        os.makedirs(self._path, exist_ok=True)

        # Setup the main folder names
        self._path_to_models  : str = os.path.join(self._path, "models")
        self._path_to_data : str = os.path.join(self._path, "data")
        self._path_to_logs : str = os.path.join(self._path, "logs")
        self._path_to_hps : str = os.path.join(self._path, "hps")


        # create the main folders
        os.makedirs(self._path_to_models, exist_ok=True)
        os.makedirs(self._path_to_data, exist_ok=True)
        os.makedirs(self._path_to_logs, exist_ok=True)
        os.makedirs(self._path_to_hps, exist_ok=True)


        # set up files paths
        self._path_to_df : str = os.path.join(self._path_to_data, "df.csv") # for dataframe
        self._path_to_pwf : str = os.path.join(self._path_to_data, "pwf.hdf5") # for pretrained word features
        self._path_to_psf : str = os.path.join(self._path_to_data, "psf.hdf5") # for pretrained segment features
        self._path_to_splits : str = os.path.join(self._path_to_data, "splits.pkl") # for splits
        self._path_to_hp_list : str = os.path.join(self._path, "hp_list.txt") # for hp history

        if not os.path.exists(self._path_to_hp_list):
            open(self._path_to_hp_list, 'w')



    @property
    def hp_ids(self):

        with open(self._path_to_hp_hist, "r") as f:
            hp_ids = [line.strip() for line in f.read().split("\n")]
        
        return hp_ids


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
                            model = self.model_name,
                            other_levels = self.other_levels,
                            evaluation_method = self.evaluation_method,
                            tasks = self.tasks,
                            subtasks =  self.subtasks,
                            all_tasks = self.all_tasks,
                            task_labels = self.task_labels,
                            prediction_level = self.prediction_level,
                            sample_level = self.sample_level,
                            input_level = self.input_level,
                            feature2dim = self.feature2dim,
                            pretrained_features = self.feature2param, 
                            )
        
        self.__check_config(config)

        config_fp = os.path.join(self._path, "config.json")
        if not os.path.exists(config_fp):
            with open(config_fp, "w") as f:
                json.dump(config, f, indent=4)

        return config


    def _check_data(self):
        return all([
                    os.path.exists(self._path_to_df),
                    True if not self._use_psf else os.path.exists(self._path_to_psf),
                    True if not self._use_pwf else os.path.exists(self._path_to_pwf)
                    ])


    def __preprocess_dataset(self, dataset:DataSet):
    
        if self._check_data():
            return None

        try:
            df = self._process_dataset(dataset)

            if self._use_psf or self._use_pwf:
                self._preprocess_pretrained_features(df)
        
        #if it breaks we delete data and have to start over to ensure nothing is corrupted
        except BaseException as e:
            shutil.rmtree(self._path_to_data)
            raise e


    def deactivate(self):
        del self.feature2model
        del self.nlp


    def activate(self):
        raise NotImplementedError

        # self.nlp = self._load_nlp_model()

        # pretrained_features = [getattr(segnlp.features, "name")(**params) in self.config["features"].items()]
        # self.feature2model = {fm.name:fm for fm in pretrained_features}

