
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
from pytorch_lightning.callbacks import ModelCheckpoint



#pytorch
import torch

#hotam
from hotam.datasets import DataSet
from hotam.preprocessing import Preprocessor
from hotam.preprocessing.dataset_preprocessor import PreProcessedDataset
from hotam.ptl.ptl_trainer_setup import setup_ptl_trainer
from hotam.ptl.ptl_base import PTLBase
from hotam import get_logger
from hotam.utils import set_random_seed, get_time, create_uid
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
                model:torch.nn.Module,
                features:list =[],
                encodings:list =[],
                model_dir:str = None,
                tokens_per_sample:bool=False,
                other_levels:list=[],
                evaluation_method:str = "default",
                root_dir:str =f"{user_dir}/.hotam/" #".hotam/pipelines"       
                ):
        
        self.project = project
        self.evaluation_method = evaluation_method
        self.model = model
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
    
        self.preprocessor = Preprocessor(                
                                        prediction_level=dataset.prediction_level,
                                        sample_level=dataset.sample_level, 
                                        input_level=dataset.level,
                                        features=features,
                                        encodings=encodings,
                                        other_levels=other_levels
                                        )

        self.dataset  = self.process_dataset(dataset)

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


        self.__eval_set = False

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

            if self.__check_for_preprocessed_data(self._path_to_data, dataset.name()):
                try:
                    logger.info(f"Loading preprocessed data from {self._path_to_data}")
                    return PreProcessedDataset(
                                                        name=dataset.name(),
                                                        dir_path=self._path_to_data,
                                                        label_encoders=self.preprocessor.encoders,
                                                        prediction_level=dataset.prediction_level
                                                        )
                except OSError as e:
                    logger.info(f"Loading failed. Will continue to preprocess data")
                    try:
                        shutil.rmtree(self._path_to_data)
                    except FileNotFoundError as e:
                        pass


            try:
                return self.preprocessor.process_dataset(dataset, dump_dir=self._path_to_data)
            except BaseException as e:
                shutil.rmtree(self._path_to_data)
                raise e


    def __check_for_preprocessed_data(self, pipeline_folder_path:str, dataset_name:str):
        fp = os.path.join(pipeline_folder_path, f"{dataset_name}_data.hdf5")
        return os.path.exists(fp)
     

    def __dump_config(self):
        config_fp = os.path.join(self._path, "config.json")
        if not os.path.exists(config_fp):
            with open(config_fp, "w") as f:
                json.dump(self.config, f, indent=4)  


    def __create_folder(self, root_dir:str, pipe_hash:str):
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


    def __get_model_args(self,
                        model:torch.nn.Module, 
                        hyperparamaters:dict,
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
        return model_args


    def __save_model_config(  self,
                            model_args:str,
                            save_choice:str, 
                            monitor_metric:str,
                            exp_model_path:str,
                            ):

        #dumping the arguments
        model_args_c = deepcopy(model_args)
        model_args_c.pop("label_encoders")
        model_args_c["model"] = model_args_c["model"].name()

        time = get_time()
        config = {
                    "time": str(time),
                    "timestamp": str(time.timestamp()),
                    "save_choice":save_choice,
                    "monitor_metric":monitor_metric,
                    "args":model_args_c,
                    }

        with open(os.path.join(exp_model_path, "model_config.json"), "w") as f:
            json.dump(config, f, indent=4)


    def eval(self):

        # if self._many_models:
        #     for model in self._trained_model:
        #         model.eval()
        # else:
        self._model.eval()
        self._model.freeze()
        self.preprocessor.deactivate_labeling()
        self.__eval_set = True


    def fit(    self,
                hyperparamaters:dict,
                exp_logger:LightningLoggerBase=None,  
                ptl_trn_args:dict={}, 
                save_choice:str = "last",  
                monitor_metric:str = "val_loss",
                ):

        model = deepcopy(self.model)
    
        if exp_logger:
            ptl_trn_args["logger"] = exp_logger
        else:
            ptl_trn_args["logger"] = None

        model_scores = []

        set_hyperparamaters = self.__create_hyperparam_sets(hyperparamaters)
        for hyperparamaters in set_hyperparamaters:

            hyperparamaters["monitor_metric"] = monitor_metric

            if "random_seed" not in hyperparamaters:
                hyperparamaters["random_seed"] = 42

            set_random_seed(hyperparamaters["random_seed"])
    
            model_unique_str = "".join(
                                            [model.name()]
                                            + list(map(str, hyperparamaters.keys()))
                                            + list(map(str, hyperparamaters.values()))
                                        )
            model_id = create_uid(model_unique_str)
            exp_model_path = os.path.join(self._path_to_models, model_id)

            if os.path.exists(exp_model_path):
                shutil.rmtree(exp_model_path)
                
            os.makedirs(exp_model_path, exist_ok=True) 

            model_args = self.__get_model_args(
                                                model=model, 
                                                hyperparamaters=hyperparamaters, 
                                                )

            self.__save_model_config(
                                    model_args=model_args,
                                    save_choice=save_choice,
                                    monitor_metric=monitor_metric,
                                    exp_model_path=exp_model_path
                                    )

            self.dataset.batch_size = hyperparamaters["batch_size"]

            if exp_logger:
                exp_logger.set_id(model_id)
                exp_logger.log_hyperparams(hyperparamaters)

                if isinstance(exp_logger, CometLogger):
                    exp_logger.experiment.add_tags([self.project, self.id])
                    exp_logger.experiment.log_others(exp_config)


            trainer, checkpoint_cb = setup_ptl_trainer( 
                                                        ptl_trn_args=ptl_trn_args,
                                                        hyperparamaters=hyperparamaters, 
                                                        exp_model_path=exp_model_path,
                                                        save_choice=save_choice, 
                                                        #prefix=model_id,
                                                        )

            get_evaluation_method(self.evaluation_method)(
                                                            model_args = model_args,
                                                            trainer = trainer,
                                                            dataset = self.dataset,
                                                            )


            if save_choice == "last":
                model_fp = checkpoint_cb.last_model_path
                checkpoint_dict = torch.load(model_fp)
                model_score = float(checkpoint_dict["callbacks"][ModelCheckpoint]["current_score"])
                
            else:
                model_fp = checkpoint_cb.best_model_path
                model_score = float(checkpoint_cb.best_model_score)

            model_scores.append({
                                "model_id":model_id, 
                                "score":model_score, 
                                "monitor_metric":monitor_metric,
                                "path":model_fp, 
                                "config_path": os.path.join(exp_model_path, "model_config.json")
                                })

        model_ranking = pd.DataFrame(model_scores)
        model_ranking.sort_values("score", ascending=False if "loss" in monitor_metric else True, inplace=True)


        with open(os.path.join(self._path_to_models,"model_rankings.json"), "w") as f:
            json.dump(model_ranking.to_dict(), f, indent=4)


    def test(   self, 
                path_to_ckpt:str=None,
                model_id:str=None,
                ptl_trn_args:dict={}
                ):


        self.dataset.split_id = 0


        with open(os.path.join(self._path_to_models,"model_rankings.json"), "r") as f:
            model_rankings = pd.DataFrame(json.load(f))
    
        if path_to_ckpt:
            ckpt_fp = path_to_ckpt
            model_config_fp = os.path.join(path_to_ckpt.split("/", 1)[0], "model_config.json")
        else:

            if model_id:
                row = model_rankings[model_rankings["model_id"] == model_id].to_dict()
            else:
                row = model_rankings.iloc[0].to_dict()

            ckpt_fp = row["path"]
            model_config_fp = row["config_path"]

        with open(model_config_fp, "r") as f:
            model_config = json.load(f)

        hyperparamaters = model_config["args"]["hyperparamaters"]

        self.dataset.batch_size = hyperparamaters["batch_size"]

        trainer, _ = setup_ptl_trainer( 
                                    ptl_trn_args=ptl_trn_args,
                                    hyperparamaters=hyperparamaters, 
                                    exp_model_path="",
                                    save_choice="", 
                                    )


        model_config["args"]["model"] = get_model(model_config["args"]["model"])
        model_config["args"]["label_encoders"] = self.preprocessor.encoders
        model_config["args"]["training"] = False
        model = PTLBase.load_from_checkpoint(ckpt_fp, **model_config["args"])

        outputs = trainer.test(
                                model=model, 
                                test_dataloaders=self.dataset.test_dataloader()
                                )

        return outputs
        



    def predict(self, doc:str):

        if not self.__eval_set:
            raise RuntimeError("Need to set pipeline to evaluation mode by using .eval() command")

        Input = self.preprocessor([doc])
        Input.sort()
        Input.to_tensor(device="cpu")

        output = self._model(Input)
        #output.chain()
        return output
        
