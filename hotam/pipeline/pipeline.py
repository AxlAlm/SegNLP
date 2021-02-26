
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
from hotam.ptl.ptl_trainer_setup import default_trainer_args
from hotam import get_logger
from hotam.utils import set_random_seed
from hotam.evaluation_methods import get_evaluation_method

logger = get_logger("PIPELINE")



class Pipeline:
    
    def __init__(self,
                name:str,
                project:str,
                tasks:List[str],
                prediction_level:str,
                sample_level:str,
                input_level:str,
                features:list =[],
                encodings:list =[],
                model_load_path:str = None,
                tokens_per_sample:bool=False,
                dataset:Union[DataSet, PreProcessedDataset] = None,
                root_dir:str = "/tmp/hotam/pipelines"       
                ):
        
        self.project = project
        self.name = name

        pipe_hash = self.__pipeline_hash(
                                            prediction_level, 
                                            sample_level, 
                                            dataset.name, 
                                            tasks, 
                                            [f.name for f in features], 
                                            encodings
                                            )                         
        pipeline_folder_path = self.__create_pipe_folder(root_dir=root_dir, pipe_hash=pipe_hash)
        self.__dump_pipe_config(
                                config = dict(
                                            prediction_level=prediction_level, 
                                            sample_level=sample_level, 
                                            dataset_name=dataset.name, 
                                            tasks=tasks, 
                                            features=[f.name for f in features], 
                                            encodings=encodings
                                            ),
                                pipeline_folder_path=pipeline_folder_path
                                )       
        
        self.preprocessor = Preprocessor(                
                                        prediction_level=prediction_level,
                                        sample_level=sample_level, 
                                        input_level=input_level,
                                        features=features,
                                        encodings=encodings,
                                        tokens_per_sample=tokens_per_sample,
                                        )

        if dataset:

            self.preprocessor.expect_labels(
                                            tasks=tasks, 
                                            task_labels=dataset.task_labels
                                            )

            if isinstance(dataset, PreProcessedDataset):
                self.dataset = dataset
            else:
                data_fp = self.__check_for_preprocessed_data(pipeline_folder_path)

                if data_fp:
                    logger.info(f"Loading preprocessed data from {data_fp}")
                    self.dataset = PreProcessedDataset(
                                                        h5py_file_path=data_fp, 
                                                        splits=dataset.splits
                                                        )
                else:
                    try:

                        self.dataset = self.preprocessor.process_dataset(dataset, dump_dir=pipeline_folder_path, chunks=5)
                    except BaseException as e:
                        shutil.rmtree(pipeline_folder_path)
                        raise e

            
        if model_load_path:
            raise NotImplementedError

        # if hyperparamaters:
            # assert model, "No model have been added"


            # self._many_models = False
            # self._nr_models = len(self._set_hyperparamaters)
            # if self._nr_models > 1:
            #     self._many_models = True


            # self._model_loaded = False
            # if model_load_path:
            #     self._model_loaded = True
            #     assert not self._many_models, "If loading a model you can continue training it but you are not allowed to trainer other models with other hyperparamaters. Make sure you hyperparamaters do not have multiple values."

                # self._trained_model = PTLBase(   
                #                                 model=model, 
                #                                 hyperparamaters=hyperparamaters,
                #                                 all_tasks=self.preprocessor.all_tasks,
                #                                 label_encoders=self.preprocessor.encoders,
                #                                 prediction_level=prediction_level,
                #                                 task_dims=task_dims,
                #                                 feature_dims=feature_dims,
                #                                 )
                # self._trained_model.load_from_checkpoint(model_load_path)


            # self._trained_model = None
            # if self._many_models:
            #     self._trained_model = []


    def __dump_pipe_config(self, config:dict, pipeline_folder_path:str):
        fp = os.path.join(pipeline_folder_path, "config.json")
        if not os.path.exists(fp):
            with open(fp, "w") as f:
                json.dump(config, f)


    def __dump_pipe_config(self, config:dict, pipeline_folder_path:str):
        fp = os.path.join(pipeline_folder_path, "config.json")
        if not os.path.exists(fp):
            with open(fp, "w") as f:
                json.dump(config, f)


    def __check_for_preprocessed_data(self, pipeline_folder_path:str):
        fp = os.path.join(pipeline_folder_path, "data.hdf5")
        if os.path.exists(fp):
            return fp
        else:
            return None


    def __pipeline_hash(self, prediction_level, sample_level, dataset_name, tasks, features, encodings):
        big_string = "%".join([prediction_level, sample_level, dataset_name] + tasks + features + encodings)
        hash_encoding = hashlib.sha224(big_string.encode()).hexdigest()
        return hash_encoding


    def __create_pipe_folder(self, root_dir:str, pipe_hash:str):
        pipeline_folder_path = os.path.join(root_dir, pipe_hash)
        os.makedirs(pipeline_folder_path, exist_ok=True)
        return pipeline_folder_path


    def __config(self, experiment_id:str, hyperparamaters:dict, evaluation_method:str, save_choice:str, model_name:str):

        exp_config = {}

        #same for whole pipeline
        exp_config["experiment_id"] = experiment_id
        exp_config["project"] = self.project
        exp_config["dataset"] = self.dataset.name
        exp_config["model"] = model_name
        exp_config["dataset_stats"] = self.dataset.stats().to_dict()
        exp_config["start_timestamp"] = get_timestamp()
        exp_config["trainer_args"] = trainer_args
        exp_config["status"] = "ongoing"

        #for each exp / model
        exp_config["hyperparamaters"] = hyperparamaters
        exp_config["evaluation_method"] = evaluation_method
        exp_config["model_selection"] = save_choice

        exp_config.update(self.preprocessor.config)

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


    # def __get_test_model_choice(self, trainer_args:dict):
    #     """we need to know, when testing, if we are selecting the last model or teh best model
    #     when testing. This simply return "last" or "best" so we know whick model the testing was done with.

    #     Parameters
    #     ----------
    #     exp_config : dict
    #         experiment configuration
    #     """

    #     test_model_choice = "last"
    #     save_top_k = trainer_args.get("save_top_k", 0)
    #     #using_callback = True if exp_config["trainer_args"]["checkpoint_callback"] else False
    #     #test_model = "best"

    #     #if save_top_k == 0:
    #         #is_saving = False

    #     if save_top_k != 0 and save_top_k:
    #         test_model_choice = "best"
        
    #     return test_model_choice


    def fit(    self,   
                model:torch.nn.Module,
                hyperparamaters:dict,
                exp_logger:LightningLoggerBase = None,  
                ptl_trn_args:dict=default_trainer_args, 
                mode:str = "all", 
                save:str = None, 
                evaluation_method:str = "default", 
                model_dump_path:str = "/tmp/hotam_models/",
                run_test:bool = True, 
                ):
        

        set_hyperparamaters = self.__create_hyperparam_sets(hyperparamaters)

        print(set_hyperparamaters)

        # if save is None:
        #     save  = self.__get_test_model_choice(ptl_trn_args)


        for hyperparamater in set_hyperparamaters:

            experiment_id = "_".join([model.name(), str(uuid.uuid4())[:8]])

            if exp_logger:
                ptl_trn_args["logger"] = exp_logger

            trainer = get_ptl_trainer( 
                                        experiment_id=experiment_id, 
                                        trainer_args=ptl_trn_args, 
                                        hyperparamaters=hyperparamater, 
                                        model_dump_path=model_dump_path,
                                        save_choice=save, 
                                        )

            if "random_seed" not in hyperparamater:
                hyperparamater["random_seed"] = 42
            
            set_random_seed(hyperparamater["random_seed"])

            # exp_config = self.__config(
            #                             experiment_id = experiment_id,
            #                             hyperparamaters = hyperparamater,
            #                             evaluation_method = evaluation_method, 
            #                             save_choice = save,
            #                             model_name=model.name()
            #                             )

            ptl_model = PTLBase(   
                                model=model, 
                                hyperparamaters=hyperparamater,
                                all_tasks=self.preprocessor.all_tasks,
                                label_encoders=self.preprocessor.encoders,
                                prediction_level=self.preprocessor.prediction_level,
                                task_dims={t:len(l) for t,l in self.preprocessor.task2labels.items()},
                                feature_dims=self.preprocessor.feature2dim,
                                )

            self.dataset.batch_size = hyperparamaters["batch_size"]

            if exp_logger:
                exp_logger.log_experiment(exp_config)

            try:
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
        
