
#basics
import uuid
from typing import List, Tuple, Dict, Callable, Union
import itertools
import json
import warnings
import numpy as np
import hashlib
import os
import shutil

#pytorch Lightning
from pytorch_lightning.loggers import LightningLoggerBase

#pytorch
import torch

#hotam
from hotam.datasets import DataSet
from hotam.preprocessing import Preprocessor
from hotam.preprocessing.dataset_preprocessor import PreProcessedDataset
from hotam.ptl import get_ptl_trainer, PTLBase
from hotam.ptl.ptl_trainer_setup import default_ptl_trn_args
from hotam import get_logger
from hotam.utils import set_random_seed, get_timestamp
from hotam.evaluation_methods import get_evaluation_method
from hotam.loggers import LocalLogger

logger = get_logger("PIPELINE")



class Pipeline:
    
    def __init__(self,
                project:str,
                tasks:List[str],
                prediction_level:str,
                sample_level:str,
                input_level:str,
                features:list =[],
                encodings:list =[],
                model_load_path:str = None,
                tokens_per_sample:bool=False,
                root_dir:str = "/tmp/hotam/pipelines"       
                ):
        
        self.tasks = tasks
        self.project = project

        pipe_hash = self.__pipeline_hash(
                                            prediction_level, 
                                            sample_level, 
                                            tasks, 
                                            [f.name for f in features], 
                                            encodings
                                            )                         
        self._pipeline_folder_path = self.__create_pipe_folder(root_dir=root_dir, pipe_hash=pipe_hash)
        self.__dump_pipe_config(
                                config = dict(
                                            prediction_level=prediction_level, 
                                            sample_level=sample_level, 
                                            tasks=tasks, 
                                            features=[f.name for f in features], 
                                            encodings=encodings
                                            ),
                                pipeline_folder_path=self._pipeline_folder_path
                                )       
        
        self.preprocessor = Preprocessor(                
                                        prediction_level=prediction_level,
                                        sample_level=sample_level, 
                                        input_level=input_level,
                                        features=features,
                                        encodings=encodings,
                                        tokens_per_sample=tokens_per_sample,
                                        )

        if model_load_path:
            raise NotImplementedError


    def process_dataset(self, dataset:Union[DataSet, PreProcessedDataset]):

        self.preprocessor.expect_labels(
                                        tasks=self.tasks, 
                                        task_labels=dataset.task_labels
                                        )

        if isinstance(dataset, PreProcessedDataset):
            return dataset
        else:

            if self.__check_for_preprocessed_data(self._pipeline_folder_path, dataset.name):
                logger.info(f"Loading preprocessed data from {self._pipeline_folder_path}")
                dataset = PreProcessedDataset(
                                                    name=dataset.name,
                                                    dir_path=self._pipeline_folder_path, 
                                                    )
            else:
                try:

                    dataset = self.preprocessor.process_dataset(dataset, dump_dir=self._pipeline_folder_path, chunks=5)
                except BaseException as e:
                    shutil.rmtree(self._pipeline_folder_path)
                    raise e
                
            return dataset


    def __dump_pipe_config(self, config:dict, pipeline_folder_path:str):
        fp = os.path.join(pipeline_folder_path, "config.json")
        if not os.path.exists(fp):
            with open(fp, "w") as f:
                json.dump(config, f)


    def __check_for_preprocessed_data(self, pipeline_folder_path:str, dataset_name:str):
        fp = os.path.join(pipeline_folder_path, f"{dataset_name}_data.hdf5")
        return os.path.exists(fp)
     

    def __pipeline_hash(self, prediction_level, sample_level, tasks, features, encodings):
        big_string = "%".join([prediction_level, sample_level] + tasks + features + encodings)
        hash_encoding = hashlib.sha224(big_string.encode()).hexdigest()
        return hash_encoding


    def __create_pipe_folder(self, root_dir:str, pipe_hash:str):
        pipeline_folder_path = os.path.join(root_dir, pipe_hash)
        os.makedirs(pipeline_folder_path, exist_ok=True)
        return pipeline_folder_path


    def __config(self, experiment_id:str, ptl_trn_args:dict, hyperparamaters:dict, evaluation_method:str, save_choice:str, model_name:str):

        config = {}

        #same for whole pipeline
        config["experiment_id"] = experiment_id
        config["project"] = self.project
        config["dataset"] = self.dataset.name
        config["model"] = model_name
        config["dataset_stats"] = self.dataset.stats.to_dict()
        config["start_timestamp"] = get_timestamp()
        config["ptl_trn_args"] = ptl_trn_args
        config["status"] = "ongoing"

        #for each exp / model
        config["hyperparamaters"] = hyperparamaters
        config["evaluation_method"] = evaluation_method
        config["model_selection"] = save_choice

        config.update(self.preprocessor.config)

        return config


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


    def fit(    self,
                model:torch.nn.Module,
                hyperparamaters:dict,
                dataset:Union[DataSet, PreProcessedDataset],
                exp_logger:LightningLoggerBase=None,  
                ptl_trn_args:dict=None, 
                save:str = "last", 
                evaluation_method:str = "default", 
                model_dump_path:str = "/tmp/hotam_models/",
                run_test:bool = True, 
                ):

        if ptl_trn_args is None:
            ptl_trn_args = default_ptl_trn_args

        if exp_logger is None:
            exp_logger = LocalLogger()

        self.dataset = self.process_dataset(dataset)
        set_hyperparamaters = self.__create_hyperparam_sets(hyperparamaters)

        for hyperparamater in set_hyperparamaters:

            experiment_id = "_".join([model.name(), str(uuid.uuid4())[:8]])

            if exp_logger:
                ptl_trn_args["logger"] = exp_logger

            trainer = get_ptl_trainer( 
                                        experiment_id=experiment_id, 
                                        ptl_trn_args=ptl_trn_args, 
                                        hyperparamaters=hyperparamater, 
                                        model_dump_path=model_dump_path,
                                        save_choice=save, 
                                        )

            if "random_seed" not in hyperparamater:
                hyperparamater["random_seed"] = 42
            
            set_random_seed(hyperparamater["random_seed"])

            exp_config = self.__config(
                                        experiment_id = experiment_id,
                                        ptl_trn_args=ptl_trn_args,
                                        hyperparamaters = hyperparamater,
                                        evaluation_method = evaluation_method, 
                                        save_choice = save,
                                        model_name=model.name()
                                        )

            ptl_model = PTLBase(   
                                model=model, 
                                hyperparamaters=hyperparamater,
                                tasks=self.preprocessor.tasks,
                                all_tasks=self.preprocessor.all_tasks,
                                label_encoders=self.preprocessor.encoders,
                                prediction_level=self.preprocessor.prediction_level,
                                task_dims={t:len(l) for t,l in self.preprocessor.task2labels.items() if t in self.preprocessor.tasks},
                                feature_dims=self.preprocessor.feature2dim,
                                )

            self.dataset.batch_size = hyperparamaters["batch_size"]

            if exp_logger:
                exp_logger.log_experiment(exp_config)

            try:

                print("Experiment is Running. Go the the dashboard to view experiment progress .. ")
                get_evaluation_method(evaluation_method)(
                                                        trainer = trainer, 
                                                        ptl_model = ptl_model,
                                                        dataset=self.dataset,
                                                        save_choice = save,
                                                        )

            except BaseException as e:
                if exp_logger:
                    exp_logger.update_config(experiment_id, key="status", value="broken")
                raise e

            if exp_logger:
                exp_logger.update_config(experiment_id, key="status", value="done")


    def eval(self):

        if self._many_models:
            for model in self._trained_model:
                model.eval()
        else:
            self._trained_model.eval()


    def predict(self, doc:Union[str,List[str]]):
        model_input = self.preprocessor(doc)
        return self._trained_model(model_input)
        
