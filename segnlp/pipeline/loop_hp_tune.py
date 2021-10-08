    
#basics
from operator import pos
from typing import List, Dict, Tuple, Union
import itertools
import json
import os
from copy import deepcopy
from numpy.core.fromnumeric import product
from numpy.lib.arraysetops import isin
from numpy.lib.shape_base import vsplit
from tqdm.auto import tqdm
from glob import glob
import shutil
import numpy as np


# pytroch
import torch


#segnlp
from segnlp import get_logger
import segnlp.utils as utils
from segnlp.utils.stat_sig import compare_dists

logger = get_logger("LOOP HP TUNING")


class HPTuneLoop:


    def __get_hyperparam_sets(self, hyperparamaters:dict) -> List[dict]:
        """

        Create a set of hyperparamaters

        """
        to_list = lambda x: [x] if not isinstance(x, list) else x
        variations = []
        for type_, type_dict in hyperparamaters.items():

            sub_type_vars = []

            for k, v in type_dict.items():

                if isinstance(v, dict):

                    subsub_type_vars = [to_list(vv) for vv in v.values()]
                    subsub_type_groups = [dict(zip(v.keys(), x)) for x in itertools.product(*subsub_type_vars)]
                    sub_type_vars.append(subsub_type_groups)

                else:
                    sub_type_vars.append(to_list(v))
            
            variations.append([dict(zip(type_dict.keys(), x)) for x in itertools.product(*sub_type_vars)])

        hp_sets = [dict(zip(hyperparamaters.keys(),v)) for v in itertools.product(*variations)]

        return hp_sets


    def __get_hyperparamaters_uid(self, set_hyperparamaters: List[dict]) -> List[Tuple[str, dict]]:
        """
        creates a unique hash id for each hyperparamater. Used to check if hyperparamaters are already tested
        """

        #give each hp an unique id
        create_hp_uid = lambda x: utils.create_uid("".join(list(map(str, x.keys())) + list(map(str, x.values()))))
        hp_uids = [create_hp_uid(hp) for hp in set_hyperparamaters]
        return list(zip(hp_uids, set_hyperparamaters))


    def __filter_and_setup_hp_configs(self, identifed_hps: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
        """
        check which hyperparamaters ids already exist and remove them  
        """

        # get all configs
        uid2config = {c["uid"]:c for c in self.hp_configs}

        hp_configs = []
        hp_id = len(uid2config)
        for hp_uid, hp in identifed_hps:


            if hp_uid in uid2config:


                if uid2config[hp_uid]["done"]:
                    continue

                config = uid2config[hp_uid]

            else:
                config = {
                        "id": str(hp_id),
                        "uid": hp_uid,
                        "hyperparamaters": deepcopy(hp),
                        "random_seeds_todo" : utils.random_ints(self.n_random_seeds),
                        "random_seeds_done" : [],
                        "path_to_models": os.path.join(self._path_to_models, str(hp_id)),
                        "done": False
                        }
                hp_id += 1

            hp_configs.append(config)

        return hp_configs


    def __dump_hp_config(self, config:dict):

        #create file for the hyperparamaters
        fp = os.path.join(self._path_to_hps, config["id"] + ".json")

        #save config
        utils.save_json(config, fp)
   

    def train(self,
                hyperparamaters:dict,
                monitor_metric:str = "val_f1",
                gpus : list = None,
                overfit_n_batches : int = None
                ):

        self.training = True

        hp_sets = self.__get_hyperparam_sets(hyperparamaters)
        uid_hps_set = self.__get_hyperparamaters_uid(hp_sets)
        hp_configs = self.__filter_and_setup_hp_configs(uid_hps_set)


        device  = "cpu"
        if gpus:      
            device =  f"cuda:{gpus[0]}"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        
        for hp_config in tqdm(hp_configs, desc="Hyperparamaters",  position=0, leave = False):    
            
            #make a folder in models for the model
            os.makedirs(hp_config["path_to_models"], exist_ok=True)

            random_seeds = hp_config["random_seeds_todo"].copy()
            for random_seed in tqdm(random_seeds, desc = "Random Seeds", position=1, leave = False):
                
                try:
                    self.fit(
                            hp_id = hp_config["id"],
                            random_seed = random_seed,
                            hyperparamaters = deepcopy(hp_config["hyperparamaters"]),
                            monitor_metric = monitor_metric,
                            device = device,
                            overfit_n_batches = overfit_n_batches
                            )

                    #update our config
                    hp_config["random_seeds_todo"].remove(random_seed)
                    hp_config["random_seeds_done"].append(random_seed)

                    # check if all random_seeds are done
                    if len(hp_config["random_seeds_todo"]) == 0:
                        hp_config["done"]

                    # write hyperparamters along with timestamps to a json
                    self.__dump_hp_config(hp_config)


                except BaseException as e:
                    
                    files_to_remove = glob(os.path.join(hp_config["path_to_models"], str(random_seed) + "*.ckpt"))
                    files_to_remove += glob(os.path.join(self._path_to_logs, str(hp_config["id"]), str(random_seed) + "*.log"))

                    for fp in files_to_remove:
                        os.remove(fp)

                    raise e


        self.rank_hps(
                            monitor_metric = monitor_metric
                        )
