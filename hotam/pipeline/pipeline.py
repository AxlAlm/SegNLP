
#basics
import uuid
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

#pytorch Lightning
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers import CometLogger


#pytorch
import torch

#hotam
from hotam.datasets import DataSet
from hotam.preprocessing import Preprocessor
from hotam.preprocessing.dataset_preprocessor import PreProcessedDataset
from hotam.ptl.ptl_trainer_setup import default_ptl_trn_args
from hotam import get_logger
from hotam.utils import set_random_seed, get_timestamp, create_uid
from hotam.evaluation_methods import get_evaluation_method
from hotam.nn.models import get_model
from hotam.features import get_feature
from hotam.nn import ModelOutput

logger = get_logger("PIPELINE")


user_dir = pwd.getpwuid(os.getuid()).pw_dir

# tasks:List[str],
# prediction_level:str,
# sample_level:str,
# input_level:str,

class Pipeline:
    
    def __init__(self,
                project:str,
                dataset:str,
                features:list =[],
                encodings:list =[],
                model_dir:str = None,
                tokens_per_sample:bool=False,
                other_levels:list=[],
                argumentative_markers:bool=False,
                root_dir:str =f"{user_dir}/.hotam/pipelines" #".hotam/pipelines"       
                ):
        
        self.project = project
        self.pipeline_id = create_uid(
                                            "".join([
                                                dataset.prediction_level,
                                                dataset.name(),
                                                dataset.sample_level, 
                                                dataset.level,
                                                ]
                                                +dataset.tasks
                                                +encodings
                                                +[f.name for f in features]
                                                    )
                                            )        
        self._pipeline_folder_path = self.__create_pipe_folder(root_dir=root_dir, pipe_hash=self.pipeline_id)
        self.config = dict(
                            project=project,
                            dataset=dataset.name(),
                            prediction_level=dataset.prediction_level, 
                            input_level=dataset.level,
                            sample_level=dataset.sample_level, 
                            tasks=dataset.tasks, 
                            features={f.name:f.params for f in features}, 
                            encodings=encodings,
                            tokens_per_sample=tokens_per_sample,
                            argumentative_markers=argumentative_markers,
                            root_dir=root_dir
                            )
        self.__dump_config()
    
        self.preprocessor = Preprocessor(                
                                        prediction_level=dataset.prediction_level,
                                        sample_level=dataset.sample_level, 
                                        input_level=dataset.level,
                                        features=features,
                                        encodings=encodings,
                                        tokens_per_sample=tokens_per_sample,
                                        argumentative_markers=argumentative_markers
                                        )


        self.dataset  = self.process_dataset(dataset)

        self.__eval_set = False
        if model_dir:
            ckpt_fp = glob(model_dir + "/*.ckpt")[0]
            args_fp = os.path.join(model_dir, "args.json")

            with open(args_fp, "r") as f:
                model_args = json.load(f)

            model_args["model"] = get_model(model_args["model"])
            model_args["label_encoders"] = self.preprocessor.encoders
            model_args["training"] = False
            self._model = PTLBase(**model_args)
            self._model = self._model.load_from_checkpoint(ckpt_fp, **model_args)


    @classmethod
    def load(self, model_dir_path:str=None, pipeline_folder:str=None, root_dir:str =f"{user_dir}/.hotam/pipelines"):
        
        if model_dir_path:

            with open(model_dir_path+"/pipeline_id.txt", "r") as f:
                pipeline_id = f.readlines()[0]

            with open(root_dir+f"/{pipeline_id}/config.json", "r") as f:
                pipeline_args = json.load(f)

            pipeline_args["model_dir"] = model_dir_path
            pipeline_args["features"] = [get_feature(fn)(**params) for fn, params in pipeline_args["features".items()]]
            return Pipeline(**pipeline_args)

 
    def process_dataset(self, dataset:Union[DataSet, PreProcessedDataset]):

        self.preprocessor.expect_labels(
                                        tasks=dataset.tasks, 
                                        subtasks=dataset.subtasks,
                                        task_labels=dataset.task_labels
                                        )

        if isinstance(dataset, PreProcessedDataset):
            pass
        else:

            if self.__check_for_preprocessed_data(self._pipeline_folder_path, dataset.name()):
                try:
                    logger.info(f"Loading preprocessed data from {self._pipeline_folder_path}")
                    return PreProcessedDataset(
                                                        name=dataset.name(),
                                                        dir_path=self._pipeline_folder_path,
                                                        label_encoders=self.preprocessor.encoders,
                                                        prediction_level=dataset.prediction_level
                                                        )
                except OSError as e:
                    logger.info(f"Loading failed. Will continue to preprocess data")
                    try:
                        shutil.rmtree(self._pipeline_folder_path)
                    except FileNotFoundError as e:
                        pass


            try:
                return self.preprocessor.process_dataset(dataset, dump_dir=self._pipeline_folder_path, chunks=5)
            except BaseException as e:
                shutil.rmtree(self._pipeline_folder_path)
                raise e


    def __check_for_preprocessed_data(self, pipeline_folder_path:str, dataset_name:str):
        fp = os.path.join(pipeline_folder_path, f"{dataset_name}_data.hdf5")
        return os.path.exists(fp)
     

    def __dump_config(self):
        config_fp = os.path.join(self._pipeline_folder_path, "config.json")
        if not os.path.exists(config_fp):
            with open(config_fp, "w") as f:
                json.dump(self.config, f)  


    def __create_pipe_folder(self, root_dir:str, pipe_hash:str):
        pipeline_folder_path = os.path.join(root_dir, pipe_hash)
        os.makedirs(pipeline_folder_path, exist_ok=True)
        return pipeline_folder_path


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


    def __get_save_model_args(  self,
                                model:torch.nn.Module, 
                                hyperparamaters:dict, 
                                exp_dump_path:str
                                ):

        model_args = dict(
                        model=model, 
                        hyperparamaters=hyperparamaters,
                        tasks=self.preprocessor.tasks,
                        all_tasks=self.preprocessor.all_tasks,
                        label_encoders=self.preprocessor.encoders,
                        prediction_level=self.preprocessor.prediction_level,
                        task_dims={t:len(l) for t,l in self.preprocessor.task2labels.items() if t in self.preprocessor.tasks},
                        feature_dims=self.preprocessor.feature2dim,
                        )

        #dumping the arguments
        model_args_c = deepcopy(model_args)
        model_args_c.pop("label_encoders")
        model_args_c["model"] = model_args_c["model"].name()
        with open(os.path.join(exp_dump_path, "args.json"), "w") as f:
            json.dump(model_args_c, f, indent=4)

        with open(os.path.join(exp_dump_path, "pipeline_id.txt"), "w") as f:
            f.write(self.pipeline_id)
    
        return model_args


    def __get_exp_config(   self,
                            model:str,
                            evaluation_method:str, 
                            monitor_metric:str,
                            experiment_id:str,
                            exp_dump_path:str,
                            ):
        config = {
                    "model":model.name(),
                    "dataset":self.dataset.name(),
                    "evaluation_method": evaluation_method,
                    "experiment_id": experiment_id,
                    "model_dump_path": exp_dump_path,
                    }
        config.update(self.config)
        config.update(self.preprocessor.config)

        return config


    def fit(    self,
                model:torch.nn.Module,
                hyperparamaters:dict,
                exp_logger:LightningLoggerBase=None,  
                ptl_trn_args:dict={}, 
                save:str = "last", 
                evaluation_method:str = "default", 
                model_dump_path:str = f"{user_dir}/.hotam/models",
                monitor_metric:str = "val_loss",
                run_test:bool = True, 
                ):

    
        if exp_logger:
            ptl_trn_args["logger"] = exp_logger
        else:
            ptl_trn_args["logger"] = None

        set_hyperparamaters = self.__create_hyperparam_sets(hyperparamaters)

        for hyperparamaters in set_hyperparamaters:

            hyperparamaters["monitor_metric"] = monitor_metric

            if "random_seed" not in hyperparamaters:
                hyperparamaters["random_seed"] = 42

            set_random_seed(hyperparamaters["random_seed"])
    
            model_unique_str = "".join(
                                            [model.name()]
                                            + list(map(str, hyperparamaters.keys()))
                                            + list(map(str, hyperparamaters.keys()))
                                        )
            experiment_id = create_uid(model_unique_str)
            exp_dump_path = os.path.join(model_dump_path, experiment_id)
            os.makedirs(exp_dump_path, exist_ok=True) 

            exp_config = self.__get_exp_config(
                                                model=model,
                                                evaluation_method=evaluation_method, 
                                                monitor_metric=monitor_metric,
                                                experiment_id=experiment_id,
                                                exp_dump_path=exp_dump_path,
                                                )
            model_args = self.__get_save_model_args(
                                                    model=model, 
                                                    hyperparamaters=hyperparamaters, 
                                                    exp_dump_path=exp_dump_path
                                                    )

            self.dataset.batch_size = hyperparamaters["batch_size"]

            if exp_logger:
                exp_logger.set_id(experiment_id)
                exp_logger.log_hyperparams(hyperparamaters)

                if isinstance(exp_logger, CometLogger):
                    exp_logger.experiment.add_tags([model.name()])
                    exp_logger.experiment.log_others(exp_config)


            get_evaluation_method(evaluation_method)(
                                                    experiment_id=experiment_id, 
                                                    ptl_trn_args = ptl_trn_args, 
                                                    model_args = model_args,
                                                    hyperparamaters=hyperparamaters, 
                                                    dataset=self.dataset,
                                                    model_dump_path=exp_dump_path,
                                                    save_choice=save, 
                                                    )


    def eval(self):

        # if self._many_models:
        #     for model in self._trained_model:
        #         model.eval()
        # else:
        self._model.eval()
        self._model.freeze()
        self.preprocessor.deactivate_labeling()
        self.__eval_set = True


    def test(self, save_choice="best"):
        pass

        #if save_choice == "last":
        #     outputs = trainer.test(
        #                             model=ptl_model, 
        #                             test_dataloaders=dataset.test_dataloader()
        #                             )
        # elif save_choice == "best":
        #     outputs = trainer.test(
        #                             model="best",
        #                             test_dataloaders=dataset.test_dataloader()
        #                             )
        # else:
        #     raise RuntimeError(f"'{save_choice}' is not an approptiate choice when testing models")

        # return outputs



    def predict(self, doc:str):

        if not self.__eval_set:
            raise RuntimeError("Need to set pipeline to evaluation mode by using .eval() command")

        Input = self.preprocessor([doc])
        Input.sort()
        Input.to_tensor(device="cpu")

        output = self._model(Input)
        #output.chain()
        return output
        
