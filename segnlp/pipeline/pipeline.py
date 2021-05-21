
#basics
import sys
from typing import List, Tuple, Dict, Callable, Union
import itertools
import json
import warnings
import numpy as np
import os
import shutil
import pwd
from copy import deepcopy
from glob import glob
import pandas as pd
from scipy import stats


#deepsig
from deepsig import aso

#pytorch
import torch

# segnlp
from .evaluation import Evaluation
from .ml import ML
from .stat_sig import StatSig
from segnlp import get_logger
from segnlp.datasets.base import DataSet
from segnlp.preprocessing import Preprocessor


# from segnlp.preprocessing.dataset_preprocessor import PreProcessedDataset
# from segnlp.ptl import get_ptl_trainer_args
# from segnlp.ptl import PTLBase
# from segnlp.utils import set_random_seed
# from segnlp.utils import get_time
# from segnlp.utils import create_uid
# from segnlp.utils import random_ints
# from segnlp.utils import ensure_numpy
# from segnlp.evaluation_methods import get_evaluation_method
# from segnlp.features import get_feature
# from segnlp.nn.utils import ModelOutput
# from segnlp.visuals.hp_tune_progress import HpProgress
# from segnlp.metrics import base_metric

logger = get_logger("PIPELINE")
user_dir = pwd.getpwuid(os.getuid()).pw_dir



class Pipeline(Evaluation, ML, StatSig):
    
    def __init__(self,
                project:str,
                dataset:str,
                model:torch.nn.Module,
                features:list =[],
                encodings:list =[],
                model_dir:str = None,
                tokens_per_sample:bool=False,
                other_levels:list=[],
                evaluation_method:str = "default",
                root_dir:str =f"{user_dir}/.segnlp/" #".segnlp/pipelines"       
                ):
        
        self.project = project
        self.evaluation_method = evaluation_method
        self.model = model
        self.exp_logger = None
        self.id = create_uid(
                            "".join([
                                    model.name(),
                                    dataset.prediction_level,
                                    dataset.name(),
                                    dataset.sample_level, 
                                    dataset.level,
                                    evaluation_method
                                    ]
                                    +dataset.tasks
                                    +encodings
                                    +[f.name for f in features]
                                    )
                                )   

        self._path = self.__create_folder(root_dir=root_dir, pipe_hash=self.id)
        self._path_to_models  = os.path.join(self._path, "models")
        self._path_to_data = os.path.join(self._path, "data")
        os.makedirs(self._path_to_models, exist_ok=True)
        os.makedirs(self._path_to_data, exist_ok=True)
        self._path_to_top_models = os.path.join(self._path_to_models, "top")
        self._path_to_tmp_models = os.path.join(self._path_to_models, "tmp")
        self._path_to_model_info = os.path.join(self._path_to_models, "model_info.json")
        self._path_to_hp_hist = os.path.join(self._path_to_models, "hp_hist.json")
        self._path_to_test_score = os.path.join(self._path_to_models, "test_scores.json")


        self.preprocessor = Preprocessor(                
                                        prediction_level=dataset.prediction_level,
                                        sample_level=dataset.sample_level, 
                                        input_level=dataset.level,
                                        features=features,
                                        encodings=encodings,
                                        other_levels=other_levels
                                        )
        self.data_module = self.preprocessor.process_dataset(dataset)

        #create and save config
        self.config = dict(
                            project=project,
                            dataset=dataset.name(),
                            model=model.name(),
                            features={f.name:f.params for f in features}, 
                            encodings=encodings,
                            other_levels=other_levels,
                            root_dir=root_dir,
                            evaluation_method=evaluation_method,
                            )

        self.config.update(self.preprocessor.config)
        self.__dump_config()

        # small hack to perserve
        self.__pp_feature_params = {f.name:f.params for f in features}
        self.__pp_encoders = self.preprocessor.encoders
        del self.preprocessor
        self.__pp_status = "inactive"

        self.__hp_tuning = False
        self.__eval_set = False


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


    def __dump_config(self):
        config_fp = os.path.join(self._path, "config.json")
        if not os.path.exists(config_fp):
            with open(config_fp, "w") as f:
                json.dump(self.config, f, indent=4)  


    def __create_folder(self, root_dir:str, pipe_hash:str):
        pipeline_folder_path = os.path.join(root_dir, pipe_hash)
        os.makedirs(pipeline_folder_path, exist_ok=True)
        return pipeline_folder_path

