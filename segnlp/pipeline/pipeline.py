
#basics
import shutil
from typing import List, Tuple, Dict, Callable, Union
import json
import numpy as np
import os
import pwd

#spacy
from spacy.language import Language


#pytorch
import torch

# segnlp
from .loop_train import TrainLoop
from .loop_test import TestLoop
from .loop_hp_tune import HPTuneLoop
from .dataset_preprocessor import DatasetPreprocessor
from .labeler import Labeler
from .text_processor import TextProcessor
from .splitter import Splitter
from .ptf_extractor import PretrainedFeatureExtractor
from segnlp import get_logger
from segnlp.datasets.base import DataSet
import segnlp.utils as utils
from segnlp import models
from segnlp.models.base import BaseModel

logger = get_logger("PIPELINE")
user_dir = pwd.getpwuid(os.getuid()).pw_dir


class Pipeline(
                TextProcessor, 
                DatasetPreprocessor,
                PretrainedFeatureExtractor,
                Labeler, 
                Splitter,
                HPTuneLoop,
                TrainLoop, 
                TestLoop,
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
        self.evaluation_method : str = evaluation_method
        self.metric : str = metric
        self.training : bool = True
        self.testing : bool = True


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
   

        # data storing / dataset preprocessing
        self._init_storage_done : bool = False

        # argumentative markers
        self.argumentative_markers : bool = False 
        if "am" in other_levels:
            self.argumentative_markers = True

            if dataset.name() == "MTC":
                self.am_extraction = "from_list"
            else:
                self.am_extraction = "pre"

        # pretrained featues
        self.feature2model : dict = {fm.name:fm for fm in pretrained_features}
        self.features : list = list(self.feature2model.keys())
        self._feature_groups : set = set([fm.group for fm in pretrained_features])
        self.feature2dim : dict = {fm.name:fm.feature_dim for fm in pretrained_features}
        self.feature2dim.update({
                                group:sum([fm.feature_dim for fm in pretrained_features if fm.group == group]) 
                                for group in self._feature_groups
                                })
        self._use_pwf : bool = "word_embs" in self._feature_groups
        self._use_psf : bool = "seg_embs" in self._feature_groups


        # preprocessing
        self._need_bio : bool = "seg" in self.subtasks
        self._labeling : bool = True
        self._removed : int = 0

        # Text Processing
        self.level2parents : dict = {
                                "token": ["sentence", "paragraph", "document"],
                                "sentence": ["paragraph", "document"],
                                "paragraph": ["document"],
                                "document": []
                                }

        self.parent2children : dict = {
                                "document": ["paragraph","sentence", "token"],
                                "paragraph": ["sentence", "token"],
                                "sentence": ["token"],
                                "token": []
                                }
        self._prune_hiers()

        # storing the current row for each level, used to fetch ids etc for lower lever data
        self._level_row_cache : dict = {}
        self.nlp : Language = self._load_nlp_model()
    

        #create and save config
        self.config = dict(
                            id = self.id,
                            dataset = dataset.name(),
                            model = self.model.name(),
                            features = {f.name:f.params for f in pretrained_features}, 
                            other_levels = other_levels,
                            root_dir = root_dir,
                            evaluation_method = evaluation_method,
                            tasks = self.tasks,
                            subtasks =  self.subtasks,
                            all_tasks = self.all_tasks,
                            task_labels = self.task_labels,
                            prediction_level = self.prediction_level,
                            sample_level = self.sample_level,
                            input_level = self.input_level,
                            feature2dim = self.feature2dim,
                            )

        #setup pipeline  root folder
        self._path : str = os.path.join(root_dir, self.id)

        if overwrite:
            logger.info(f"Overriding all data in {self._path} by moving existing folder to /tmp/ and creating a new folder")

            new_loc = os.path.join("/tmp/", self.id)
            if os.path.exists(new_loc):
                shutil.rmtree(new_loc)

            shutil.move(self._path, new_loc)

        os.makedirs(self._path, exist_ok=True)

        # we need to check that the previous config is the same as the current
        # otherwise we will have errors down the line
        self.__check_config()


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
        self._path_to_df : str = os.path.join(self._path_to_data, "df.hdf5") # for dataframe
        self._path_to_pwf : str = os.path.join(self._path_to_data, "pwf.hdf5") # for pretrained word features
        self._path_to_psf : str = os.path.join(self._path_to_data, "psf.hdf5") # for pretrained segment features
        self._path_to_splits : str = os.path.join(self._path_to_data, "splits.pkl") # for splits
        self._path_to_hp_hist : str = os.path.join(self._path_to_hps, "hist.txt") # for hp history

        if not os.path.exists(self._path_to_hp_hist):
            open(self._path_to_hp_hist, 'w')

        self._path_to_hp_json : str = os.path.join(self._path_to_hps, "hps.json") # for storing hps

        #dump config
        self.__dump_config()

        #processed the data
        self._preprocess_dataset(dataset)

        # create split indexes 
        self._set_splits(
                        premade_splits = dataset.splits,
                        )

        # after we have processed data we will deactivate preprocessing so we dont keep
        # large models only using in preprocessing in memory
        self.deactivate_preprocessing()

    
    @classmethod
    def load(self, model_dir_path:str=None, pipeline_folder:str=None, root_dir:str =f"{user_dir}/.segnlp/pipelines"):
        
        if model_dir_path:

            with open(model_dir_path+"/pipeline_id.txt", "r") as f:
                pipeline_id = f.readlines()[0]

            with open(root_dir+f"/{pipeline_id}/config.json", "r") as f:
                pipeline_args = json.load(f)

            pipeline_args["model_dir"] = model_dir_path
            pipeline_args["features"] = [get_feature(fn)(**params) for fn, params in pipeline_args["features".items()]]
            return Pipeline(**pipeline_args)


    def __check_config(self):

        # create a key for the id
        config_key = utils.create_uid(str(self.config))

        key_file = os.path.join(self._path, "key.txt")

        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                key = key_file.read().strip()

            if key != config_key:
                raise RuntimeError(f"Current config is not the same as the config found in {self._path}. Either change the id of the pipeline or make sure all the paramaters the same as for {self.id}")
        # else:
        #     raise RuntimeError("pipeline is missing key.txt file. Recreate the the pipeline.")


    def __dump_config(self):
        config_fp = os.path.join(self._path, "config.json")
        if not os.path.exists(config_fp):
            with open(config_fp, "w") as f:
                json.dump(self.config, f, indent=4)


    def deactivate_preprocessing(self):
        del self.feature2model
        del self.nlp


    def activate_preprocessing(self):

        self.nlp = self._load_nlp_model()

        pretrained_features = [getattr(segnlp.features, "name")(**params) in self.config["features"].items()]
        self.feature2model = {fm.name:fm for fm in pretrained_features}


    # def eval(self):

    #     # if self._many_models:
    #     #     for model in self._trained_model:
    #     #         model.eval()
    #     # else:
    #     self._model.eval()
    #     self._model.freeze()
    #     self._model.inference = True
    #     self.preprocessor.deactivate_labeling()
    #     self.__eval_set = True

