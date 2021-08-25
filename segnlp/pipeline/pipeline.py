
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
from .evaluator import Evaluator
from .trainer import Trainer
from .tester import Tester
from .stat_sig import StatSig
from .dataset_preprocessor import DatasetPreprocessor
from .encoder import Encoder
from .labeler import Labeler
from .doc_processor import DocProcessor
from .text_processor import TextProcessor
from .splitter import Splitter
from segnlp import get_logger
from segnlp.datasets.base import DataSet
import segnlp.utils as utils
from segnlp import models


logger = get_logger("PIPELINE")
user_dir = pwd.getpwuid(os.getuid()).pw_dir


class Pipeline(
                DocProcessor, 
                TextProcessor, 
                Labeler, 
                DatasetPreprocessor,
                Encoder,
                Evaluator, 
                Trainer, 
                Tester,
                StatSig,
                Splitter,
                ):
    
    def __init__(self,
                id:str,
                dataset:Union[str, DataSet],
                model:Union[torch.nn.Module, str],
                metric:str,
                pretrained_features:list = [],
                encodings:list = [],
                other_levels:list = [],
                evaluation_method:str = "default",
                root_dir:str =f"{user_dir}/.segnlp/", #".segnlp/pipelines"  
                override: bool = False,
                vocab : Union[str,list, int] = "bnc", #if int it will be the size of the vocab created from most common word from bnc
                ):

        #general
        self.id = id
        self.vocab = vocab
        self.model = getattr(models, model) if isinstance(model,str) else model
        self.evaluation_method = evaluation_method
        self.metric = metric

        # task info
        self.prediction_level = dataset.prediction_level
        self.sample_level = dataset.sample_level
        self.input_level = dataset.level
        self.tasks = dataset.tasks
        self.subtasks = dataset.subtasks
        self.task_labels = dataset.task_labels
        self.all_tasks = sorted(set(self.tasks + self.subtasks))
   

        # data storing / dataset preprocessing
        self._init_storage_done = False

        # argumentative markers
        self.argumentative_markers = False 
        if "am" in other_levels:
            self.argumentative_markers = True

            if dataset.name() == "MTC":
                self.am_extraction = "from_list"
            else:
                self.am_extraction = "pre"


        #encodings
        self.encoders = {}
        self.encodings = encodings
        self._need_deps = True if "deprel" in encodings else False
        self._create_data_encoders()
        self._create_label_encoders()
        self.label_encoders = {k:v for k,v in self.encoders if k in self.all_tasks}


        # pretrained featues
        self.feature2model = {fm.name:fm for fm in pretrained_features}
        self.features = list(self.feature2model.keys())
        self._feature_groups = set([fm.group for fm in pretrained_features])
        self.feature2dim = {fm.name:fm.feature_dim for fm in pretrained_features}
        self.feature2dim.update({
                                group:sum([fm.feature_dim for fm in pretrained_features if fm.group == group]) 
                                for group in self._feature_groups
                                })

        # preprocessing
        self._need_bio = "seg" in self.subtasks
        self._labeling = True
        self._removed = 0

        # Text Processing
        self.level2parents = {
                                "token": ["sentence", "paragraph", "document"],
                                "sentence": ["paragraph", "document"],
                                "paragraph": ["document"],
                                "document": []
                                }

        self.parent2children = {
                                "document": ["paragraph","sentence", "token"],
                                "paragraph": ["sentence", "token"],
                                "sentence": ["token"],
                                "token": []
                                }
        self._prune_hiers()

        # storing the current row for each level, used to fetch ids etc for lower lever data
        self._level_row_cache = {}
        self.nlp = self._load_nlp_model()
    

        #create and save config
        self.config = dict(
                            id = self.id,
                            dataset = dataset.name(),
                            model = self.model.name(),
                            features = {f.name:f.params for f in pretrained_features}, 
                            encodings = encodings,
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
                            encoding = self.encodings
                            )

        #setup pipeline  root folder
        self._path = os.path.join(root_dir, self.id)

        if override:
            logger.info(f"Overriding all data in {self._path} by moving existing folder to /tmp/ and creating a new folder")
            shutil.move(self._path, os.path.join("/tmp/", self.id))

        os.makedirs(self._path, exist_ok=True)

        # we need to check that the previous config is the same as the current
        # otherwise we will have errors down the line
        self.__check_config()


        #setup all all folder and file names
        self._path_to_models  = os.path.join(self._path, "models")
        self._path_to_data = os.path.join(self._path, "data")
        os.makedirs(self._path_to_models, exist_ok=True)
        os.makedirs(self._path_to_data, exist_ok=True)
        self._path_to_preprocessed_data = os.path.join(self._path_to_data, "data.hdf5")
        self._path_to_splits = os.path.join(self._path_to_data, "splits.pkl")
        self._path_to_top_models = os.path.join(self._path_to_models, "top")
        self._path_to_tmp_models = os.path.join(self._path_to_models, "tmp")
        self._path_to_model_info = os.path.join(self._path_to_models, "model_info.json")
        self._path_to_hp_hist = os.path.join(self._path_to_models, "hp_hist.json")
        self._path_to_test_score = os.path.join(self._path_to_models, "test_scores.json")

        #dump config
        self.__dump_config()

        #processed the data
        self._preprocess_dataset(dataset)

        # create split indexes 
        self._set_splits(
                        dataset = dataset,
                        evaluation_method = evaluation_method
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

