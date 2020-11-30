
#basic
from __future__ import annotations
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
import webbrowser

#hotam
from hotam.manager.ptl_trainer_setup import Trainer
from hotam.manager.evaluator import Evaluator
from hotam.utils import get_timestamp
from hotam import get_logger

#pytroch
import torch


logger = get_logger("EXPERIMENT MANAGER")

warnings.simplefilter('ignore')

default_seed = 42


def set_random_seed(nr, using_gpu:bool=False):
    random.seed(nr)
    np.random.seed(nr)
    torch.manual_seed(nr)

    if using_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_config():
    # ADD JSON SCHEMA CHECK
    pass


class ExperimentManager(Evaluator, Trainer):

    def __init__(self, exp_logger=None):
        self.exp_logger = exp_logger
    

    def __get_test_model_choice(self, trainer_args:dict):
        """we need to know, when testing, if we are selecting the last model or teh best model
        when testing. This simply return "last" or "best" so we know whick model the testing was done with.

        Parameters
        ----------
        exp_config : dict
            experiment configuration
        """

        test_model_choice = "last"
        save_top_k = trainer_args.get("save_top_k", 0)
        #using_callback = True if exp_config["trainer_args"]["checkpoint_callback"] else False
        #test_model = "best"

        #if save_top_k == 0:
            #is_saving = False

        if save_top_k != 0 and save_top_k:
            test_model_choice = "best"
        
        return test_model_choice


    def __create_hyperparam_sets(self, hyperparamaters:Dict[str,Union[str, int, float, list]]) -> Union[dict,List[dict]]:
        """creates a set of hyperparamaters for hyperparamaters based on given hyperparamaters lists.
        takes a hyperparamaters and create a set of new paramaters given that any
        paramater values are list of values.

        Parameters
        ----------
        hyperparamaters : Dict[str,Union[str, int, float, list]]
            dict of hyperparamaters.

        Returns
        -------
        Union[dict,List[dict]]
            returns a list of hyperparamaters if any hyperparamater value is a list, else return 
            original hyperparamater
        """
        hyperparamaters_reformat = {k:[v] if not isinstance(v,list) else v for k,v in hyperparamaters.items()}
        hypam_values = list(itertools.product(*list(hyperparamaters_reformat.values())))
        set_hyperparamaters = [dict(zip(list(hyperparamaters_reformat.keys()),h)) for h in hypam_values]

        return set_hyperparamaters


    def run(    self,
                project:str,
                dataset:None,
                model:None,
                hyperparamaters:dict,
                trainer_args:dict,
                monitor_metric:str,
                progress_bar_metrics:list,
                metrics_configs:list=[],
                eval_method:str="default",
                model_save_path:str=None,
                run_test:bool=False,
                verbose:int=1,
                debug_mode:bool=False,
                ):

        if debug_mode: #or trainer_args.get("overfit_batches", False):
            self.exp_logger = None     
        
        self.dataset = dataset
        self.evaluation_method = eval_method
            
        set_hyperparamaters = self.__create_hyperparam_sets(hyperparamaters)
        logger.info(f"Generated {len(set_hyperparamaters)} hyperparamater sets.")

        for hyperparamaters in set_hyperparamaters:
            
            start_ts = get_timestamp()

            model_name = model.name()
            experiment_id = "_".join([model_name, str(uuid.uuid4())[:10]])
            logger.info(f"Current Experiment ID : {experiment_id}")

            random_seed = hyperparamaters.get("random_seed", default_seed)
            set_random_seed(    
                            random_seed, 
                            using_gpu=True if hyperparamaters.get("gpus", None) else False,
                            )
            
            if self.exp_logger:
                self.exp_logger.set_exp_id(experiment_id)


            #setup Pytroch Lightning Trainer
            trainer_args_copy = deepcopy(trainer_args)
            trainer_args_copy["logger"] = self.exp_logger
            trainer = self._setup_ptl_trainer( 
                                                experiment_id = experiment_id,
                                                trainer_args = trainer_args_copy,
                                                hyperparamaters = hyperparamaters,
                                                model_save_path= model_save_path,
                                                run_test=run_test
                                                )


            # scorer = Metrics(
            #                 dataset=self.dataset,
            #                 metrics=metrics_configs, 
            #                 monitor_metric=monitor_metric, 
            #                 progress_bar_metrics=progress_bar_metrics
            #                 )
            
            save_model_choice = None
            if run_test:
                save_model_choice = self.__get_test_model_choice(trainer_args_copy)


            exp_config = {}
            exp_config["project"] = project
            exp_config["dataset"] = dataset.name
            exp_config["model"] = model_name
            exp_config["dataset_config"] = self.dataset.config
            exp_config["dataset_stats"] = self.dataset.stats().to_dict()
            exp_config["task2label"] = self.dataset.task2labels
            exp_config["tasks"] = self.dataset.tasks
            exp_config["evaluation_method"] = eval_method
            exp_config["hyperparamaters"] = hyperparamaters
            exp_config["experiment_id"] = experiment_id
            exp_config["start_timestamp"] = start_ts
            exp_config["model_selection"] = save_model_choice
            exp_config["metrics"] = scorer.metric_names
            exp_config["progress_bar_metrics"] = progress_bar_metrics
            exp_config["model_save_path"] = model_save_path
            exp_config["hyperparamaters"]["monitor_metric"] = monitor_metric
            exp_config["hyperparamaters"]["random_seed"] = random_seed
            exp_config["trainer_args"] = trainer_args

            self.dataset.batch_size = hyperparamaters["batch_size"]

            if self.exp_logger:
                self.exp_logger.log_experiment(exp_config)


            # starting dashboard app
            dashboard_app.run_server(
                                    port=8050,
                                    debug=debug_mode
                                    )
            webbrowser.open("http://127.0.0.1:8050/")



            #run training
            self._get_eval_method()(
                                    trainer=trainer, 
                                    model=model,
                                    hyperparamaters=hyperparamaters,
                                    metrics=metrics_configs, 
                                    monitor_metric=monitor_metric, 
                                    progress_bar_metrics=progress_bar_metrics,
                                    save_model_choice=save_model_choice,
                                    run_test=run_test,
                                    )
    

            #exp_config["end_timestamp"] = get_timestamp()

        logger.info("Experiments done")


        if hasattr(self.db, "close"):
            self.db.close()