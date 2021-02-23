
#basic
#from __future__ import annotations
import pandas as pd
import os
from glob import glob
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Union
import itertools
import json
import warnings
from IPython.display import display
import numpy as np
import random
from copy import deepcopy
#import webbrowser
import time
import sys

#hotam
from hotam.utils import get_timestamp
from hotam import get_logger
#from hotam.default_hyperparamaters import get_default_hps

#pytroch
#import torch


€logger = get_logger("EXPERIMENT MANAGER")
#warnings.simplefilter('ignore')


class ChainedPipeline:


    def __init__(self, pipelines=Union[List, Pipeline]):
        self.pipelines = pipelines


    def fit(self, ):

        for pipeline in self.pipelines:
            pipeline.fit(
                        exp_logger:LightningLoggerBase,  
                        ptl_trn_args:dict=default, 
                        mode:str = "all", 
                        save_model_choice:str = None, 
                        evaluation_method:str = "default", 
                        run_test:bool = True, 
                        )
                    
    def eval(self):
        for pipeline in self.pipelines:
            pipeline.eval()


    def predict(self, doc:Union[str,List[str]], chain_predictions:bool = True):
        
        for pipeline in self.pipelines:
            out = pipeline.predict(doc)

            #overwrite the doc so input to the next pipeline is output of the previuos pipeline
            if chain_predictions:
                doc = out
        
        #return output of last pipeline
        return out
        






#     def run(    self,
#                 project:str,
#                 dataset:None,
#                 model:None,
#                 monitor_metric:str,
#                 progress_bar_metrics:list,
#                 hyperparamaters:dict={},
#                 trainer_args:dict={},
#                 exp_logger=None,
#                 metrics_configs:list=[],
#                 eval_method:str="default",
#                 model_save_path:str=None,
#                 run_test:bool=False,
#                 verbose:int=1,
#                 debug_mode:bool=False,
#                 ):

#         if debug_mode:
#             self.exp_logger = None
    
#         self.dataset = dataset
#         self.evaluation_method = eval_method

#         model_name = model.name()
#         if not hyperparamaters:
#             hyperparamaters = get_default_hps(model_name)

#         if not trainer_args:
#             trainer_args = default_trainer_args

#         set_hyperparamaters = self.__create_hyperparam_sets(hyperparamaters)
#         logger.info(f"Generated {len(set_hyperparamaters)} hyperparamater sets.")

#         for hyperparamaters in set_hyperparamaters:
            
#             start_ts = get_timestamp()
#             experiment_id = "_".join([model_name, str(uuid.uuid4())[:8]])

#             logger.info(f"Current Experiment ID : {experiment_id}")
#             random_seed = hyperparamaters.get("random_seed", default_seed)
#             set_random_seed(    
#                             random_seed, 
#                             using_gpu=True if hyperparamaters.get("gpus", None) else False,
#                             )
            
#             if exp_logger:
#                 exp_logger.set_exp_id(experiment_id)
#                 assert exp_logger.experiment_id == experiment_id

#             #setup Pytroch Lightning Trainer
#             trainer_args_copy = deepcopy(trainer_args)
#             trainer_args_copy["logger"] = exp_logger
#             trainer = get_ptl_trainer( 
#                                                 experiment_id = experiment_id,
#                                                 trainer_args = trainer_args_copy,
#                                                 hyperparamaters = hyperparamaters,
#                                                 model_save_path= model_save_path,
#                                                 run_test=run_test
#                                                 )
            
#             save_model_choice = None
#             if run_test:
#                 save_model_choice = self.__get_test_model_choice(trainer_args_copy)

#             exp_config = {}
#             exp_config["project"] = project
#             exp_config["dataset"] = dataset.name
#             exp_config["model"] = model_name
#             exp_config["dataset_config"] = self.dataset.config
#             exp_config["dataset_stats"] = self.dataset.stats().to_dict()
#             exp_config["task2label"] = self.dataset.task2labels
#             exp_config["tasks"] = self.dataset.tasks
#             exp_config["subtasks"] = self.dataset.subtasks
#             exp_config["evaluation_method"] = eval_method
#             exp_config["hyperparamaters"] = hyperparamaters
#             exp_config["experiment_id"] = experiment_id
#             exp_config["start_timestamp"] = start_ts
#             exp_config["model_selection"] = save_model_choice
#             exp_config["metrics"] = Metrics.metrics()[1]
#             exp_config["progress_bar_metrics"] = progress_bar_metrics
#             exp_config["model_save_path"] = model_save_path
#             exp_config["hyperparamaters"]["monitor_metric"] = monitor_metric
#             exp_config["hyperparamaters"]["random_seed"] = random_seed
#             exp_config["trainer_args"] = trainer_args
#             exp_config["status"] = "ongoing"

#             self.dataset.batch_size = hyperparamaters["batch_size"]

#             if exp_logger:
#                 exp_logger.log_experiment(exp_config)

#             error = None
#             try:
#                 self._get_eval_method()(
#                                         trainer=trainer, 
#                                         model=model,
#                                         hyperparamaters=hyperparamaters,
#                                         metrics=metrics_configs, 
#                                         monitor_metric=monitor_metric, 
#                                         progress_bar_metrics=progress_bar_metrics,
#                                         save_model_choice=save_model_choice,
#                                         run_test=run_test,
#                                         )
            
#             except BaseException as e:
#                 if exp_logger:
#                     exp_logger.update_config(experiment_id, key="status", value="broken")
#                 raise e

#             if exp_logger:
#                 exp_logger.update_config(experiment_id, key="status", value="done")

