    
#basics
from typing import List, Dict, Tuple, Union
import itertools
import numpy as np
import json
import os
from copy import deepcopy
import shutil
import pandas as pd
from tqdm import tqdm

#pytorch
import torch

#segnlp
from segnlp import get_logger
import segnlp.utils as utils
from segnlp.utils import get_ptl_trainer_args


logger = get_logger("ML")


class TrainingUtils:


    def __create_hyperparam_sets(self, hyperparamaters:dict) -> List:
        """creates a set of hyperparamaters for hyperparamaters based on given hyperparamaters lists.
        takes a hyperparamaters and create a set of new paramaters given that any
        paramater values are list of values.
        """
        group_variations = []
        for _, gdict in hyperparamaters.items():
            vs = [[v] if not isinstance(v, list) else v for v in gdict.values()]
            group_sets = [dict(zip(gdict.keys(),v)) for v in itertools.product(*vs)]
            group_variations.append(group_sets)

        
        hp_sets = [dict(zip(hyperparamaters.keys(),v)) for v in itertools.product(*group_variations)]

        #give each hp an unique id
        create_hp_uid = lambda x: utils.create_uid("".join(list(map(str, x.keys()))+ list(map(str, x.values()))))
        hp_uids = [create_hp_uid(hp) for hp in hp_sets]


        # add ids
        hp_dicts = dict(zip(hp_uids,hp_sets))
        

        #### check if the hp_uids are already tested and logged
        with open(self._path_to_hp_hist, "r") as f:
            done_hp_ids = set(f.read().split("\n"))


        for hp_id in list(hp_dicts.keys()):
            if hp_id in done_hp_ids:
                logger.info(f"WARNING! Hyperparamater with id {hp_id} has already been run. Will remove from hyperparamter tuning.")
                hp_dicts.pop(hp_id)

    
        return hp_dicts


    def __save_model_config(  self,
                            model_args:str,
                            save_choice:str, 
                            monitor_metric:str,
                            exp_model_path:str,
                            ):

        #dumping the arguments
        model_args_c = deepcopy(model_args)

        # fix objects
        for t, hps in model_args_c["hyperparamaters"].items():
                for k,v in hps.items():
                    if not isinstance(v, (str, int, float)):
                        model_args_c["hyperparamaters"][t][k] = str(model_args_c["hyperparamaters"][t][k])
            
        model_args_c.pop("label_encoder")
        time = utils.get_time()
        config = {
                    "time": str(time),
                    "timestamp": str(time.timestamp()),
                    "save_choice":save_choice,
                    "monitor_metric":monitor_metric,
                    "args":model_args_c,
                    }

        with open(os.path.join(exp_model_path, "model_config.json"), "w") as f:
            json.dump(config, f, indent=4)


    def _fit(    self,
                model_id :str,
                hyperparamaters:dict,
                random_seed:int = 42,
                monitor_metric:str = "val_f1",
                ):

        # if model_id is None:
        #     model_id = utils.create_uid("".join(list(map(str, hyperparamaters.keys())) + list(map(str, hyperparamaters.values()))))

        utils.set_random_seed(random_seed)

        # hyperparamaters["general"]["random_seed"] = random_seed
        # hyperparamaters["general"]["monitor_metric"] = monitor_metric

        #loading our preprocessed dataset
        data_module = utils.DataModule(
                                path_to_data = self._path_to_data,
                                batch_size = hyperparamaters["general"]["batch_size"],
                                )
        
        model_save_path = os.path.join(self._path_to_models, model_id, str(random_seed))


   
        # model_args = dict(
        #                 hyperparamaters = hyperparamaters,
        #                 label_encoder = self.label_encoder,
        #                 feature_dims = self.config["feature2dim"],
        #                 metric = self.metric
        #                 )

        self.__save_model_config(
                                model_args=model_args,
                                save_choice=save_choice,
                                monitor_metric=monitor_metric,
                                exp_model_path=exp_model_path
                                )


        model_fp, model_score = self._eval(
                                            model_args = model_args,
                                            ptl_trn_args = ptl_trn_args,
                                            data_module = data_module,
                                            )



    def train(self,
                    hyperparamaters:dict,
                    ptl_trn_args:dict={},
                    n_random_seeds:int=6,
                    random_seed:int=None,
                    save_choice:str="best",
                    monitor_metric:str = "val_f1",
                    ss_test:str="aso",
                    override:bool=False
                    ):

        hp_dicts = self.__create_hyperparam_sets(hyperparamaters)
        
        print("_" * (os.get_terminal_size().columns - 3))

        for hp_uid, hp_dict in tqdm(hp_dicts.items(), desc="Hyperparamaters",  position=0):    
        
            if random_seed is not None and isinstance(random_seed, int):
                random_seeds = [random_seed]
            else:
                random_seeds = utils.random_ints(n_random_seeds)

            for random_seed in tqdm(random_seeds, start=1, desc = "Random Seeds", position=1):
                
                utils.set_random_seed(random_seed)

                self.fit(
                            hyperparamaters = hp_dict["hyperparamaters"],
                            random_seed = random_seed,
                            monitor_metric = monitor_metric,
                            model_id = hp_uid,
                            )


            #### check if the hp_uids are already tested and logged
            with open(self._path_to_hp_hist, "a") as f:
                f.write(hp_uid + "\n")

            #### check if the hp_uids are already tested and logged
            with open(self._path_to_hps, "a") as f:
                f.write(json.dumps(hp_dict) + "\n")
