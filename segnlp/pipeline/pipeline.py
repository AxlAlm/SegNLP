
#basics
from typing import List, Tuple, Dict, Callable, Union
import json
import numpy as np
import os
import pwd
import pandas as pd
from scipy import stats

#pytorch
import torch

# segnlp
from .evaluation import Evaluation
from .ml import ML
from .stat_sig import StatSig
from segnlp import get_logger
from segnlp.datasets.base import DataSet
from segnlp.preprocessing import Preprocessor
import segnlp.utils as utils
from segnlp import models

logger = get_logger("PIPELINE")
user_dir = pwd.getpwuid(os.getuid()).pw_dir



class Pipeline(Evaluation, ML, StatSig):
    
    def __init__(self,
                id:str,
                dataset:Union[str, DataSet],
                model:Union[torch.nn.Module, str],
                metric:str,
                features:list = [],
                encodings:list = [],
                other_levels:list=[],
                evaluation_method:str = "default",
                root_dir:str =f"{user_dir}/.segnlp/" #".segnlp/pipelines"       
                ):

        self.id = id
        
        if isinstance(model, str):
            model = getattr(models, model)
        
        self.model = model
        self.evaluation_method = evaluation_method
        self.metric = metric

        self.preprocessor = Preprocessor(                
                                        prediction_level=dataset.prediction_level,
                                        sample_level=dataset.sample_level, 
                                        input_level=dataset.level,
                                        tasks=dataset.tasks,
                                        subtasks=dataset.subtasks,
                                        task_labels=dataset.task_labels,
                                        features=features,
                                        encodings=encodings,
                                        other_levels=other_levels
                                        )

        #create and save config
        self.config = dict(
                            id=self.id,
                            dataset=dataset.name(),
                            model=self.model.name(),
                            features={f.name:f.params for f in features}, 
                            encodings=encodings,
                            other_levels=other_levels,
                            root_dir=root_dir,
                            evaluation_method=evaluation_method,
                            )
        self.config.update(self.preprocessor.config)


        #setup pipeline  root folder
        self._path = os.path.join(root_dir, self.id)
        os.makedirs(self._path, exist_ok=True)

        # we need to check that the previous config is the same as the current
        # otherwise we will have errors down the line
        self.__check_config()


        #setup all all folder and file names
        self._path_to_models  = os.path.join(self._path, "models")
        self._path_to_data = os.path.join(self._path, "data")
        os.makedirs(self._path_to_models, exist_ok=True)
        os.makedirs(self._path_to_data, exist_ok=True)
        self._path_to_top_models = os.path.join(self._path_to_models, "top")
        self._path_to_tmp_models = os.path.join(self._path_to_models, "tmp")
        self._path_to_model_info = os.path.join(self._path_to_models, "model_info.json")
        self._path_to_hp_hist = os.path.join(self._path_to_models, "hp_hist.json")
        self._path_to_test_score = os.path.join(self._path_to_models, "test_scores.json")

        #dump config
        self.__dump_config()


        #processed the data
        self.data_module = self.preprocessor.process_dataset(
                                                            dataset, 
                                                            dump_dir=self._path_to_data,
                                                            evaluation_method=self.evaluation_method
                                                            )


        # small hack to perserve
        self._pp_feature_params = {f.name:f.params for f in features}
        self._pp_label_encoders = {k:v for k,v in self.preprocessor.encoders.items() if k in self.config["all_tasks"]}
        del self.preprocessor
        self._pp_status = "inactive"

        self._hp_tuning = False
        self._eval_set = False
    
        self.exp_logger = None


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



