    
#basics
from typing import List, Dict, Tuple, Union
import itertools
import json
import os
from copy import deepcopy
from numpy.lib.arraysetops import isin
from tqdm.auto import tqdm
from glob import glob
import shutil


# pytroch
import torch


#segnlp
from segnlp import get_logger
import segnlp.utils as utils


logger = get_logger("LOOP HP TUNING")


class HPTuneLoop:


    def __create_hyperparam_sets(self, hyperparamaters:dict) -> List[dict]:
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


    def __id_hyperparamaters(self, set_hyperparamaters: List[dict]) -> List[Tuple[str, dict]]:
        """
        creates a unique hash id for each hyperparamater. Used to check if hyperparamaters are already tested
        """

        #give each hp an unique id
        create_hp_id = lambda x: utils.create_uid("".join(list(map(str, x.keys())) + list(map(str, x.values()))))
        hp_ids = [create_hp_id(hp) for hp in set_hyperparamaters]
        return list(zip(hp_ids, set_hyperparamaters))


    def __filter_hyperparamaters(self, identifed_hps: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
        """
        check which hyperparamaters ids already exist and remove them  
        """
        
        # hp ids for hyperparamaters that is already tested
        done_hp_ids = set([fp.replace(".json", "") for fp in os.listdir(self._path_to_hps)])

        # filtered hyperparamaters
        filtered_hps = [(id,hps) for id,hps in identifed_hps if id not in done_hp_ids]
    
        return filtered_hps


    def __dump_hps( self,
                    hp_id:str,
                    hyperparamaters: dict, 
                    random_seeds: list,
                    ):

        hyperparamaters = deepcopy(hyperparamaters)
   
        time = utils.get_time()
        config = {
                    "hp_id": hp_id,
                    "time": str(time),
                    "timestamp": str(time.timestamp()),
                    "hyperparamaters": hyperparamaters,
                    "random_seeds" : random_seeds
                    }

        #create file for the hyperparamaters
        fp = os.path.join(self._path_to_hps, hp_id + ".json")
        
        #### check if the hp_ids are already tested and logged
        with open(fp, "a") as f:
            json.dump(config, f, indent=4)



    def get_score_dists(self, monitor_metric: str):
        
        score_dists = {}
        log_dfs = self._load_logs()
        for hp_id in self.hp_ids:

            df = log_dfs[hp_id]
            df.set_index(["split"], inplace = True)
            top_scores = df.loc["val"].groupby("random_seed")[monitor_metric].max()
            score_dists[hp_id] = top_scores

        return score_dists


    def find_best_hp(self, monitor_metric:str):
        score_dists = self.get_score_dists(monitor_metric = monitor_metric)

        print(score_dists)


    def train(self,
                hyperparamaters:dict,
                n_random_seeds:int=6,
                random_seed:int=None,
                monitor_metric:str = "val_f1",
                gpus : list = [],
                overfit_n_batches : int = None
                ):

        self.training = True

        hp_sets = self.__create_hyperparam_sets(hyperparamaters)
        id_hps_set = self.__id_hyperparamaters(hp_sets)
        id_hps_set = self.__filter_hyperparamaters(id_hps_set)

        device  = "cpu"
        if gpus:      
            device =  f"cuda:{gpus[0]}"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        
        for hp_id, hps in tqdm(id_hps_set, desc="Hyperparamaters",  position=0):    
            
            #make a folder in models for the model
            path_to_hp_models = os.path.join(self._path_to_models, hp_id)
            os.makedirs(path_to_hp_models, exist_ok=True)

            if random_seed is not None and isinstance(random_seed, int):
                random_seeds = [random_seed]
            else:
                random_seeds = utils.random_ints(n_random_seeds)

            #done_random_seeds = []
            for random_seed in tqdm(random_seeds, desc = "Random Seeds", position=1):
                
                try:
                    self.fit(
                            hp_id = hp_id,
                            random_seed = random_seed,
                            hyperparamaters = deepcopy(hps),
                            monitor_metric = monitor_metric,
                            device = device,
                            overfit_n_batches = overfit_n_batches
                            )
                except BaseException as e:
                    # failed_models = glob(os.path.join(path_to_hp_models, random_seed + "*.ckpt"))
                    # for fp in failed_models:
                    #     os.remove(fp)

                    shutil.rmtree(path_to_hp_models)
                    raise e
                
                #done_random_seeds.append(random_seed)
 
 
            # write hyperparamters along with timestamps to a json
            self.__dump_hps(
                            hp_id = hp_id, 
                            hyperparamaters = hps,
                            random_seeds = random_seeds
                            )
        


        self.find_best_hp(
                            monitor_metric = monitor_metric
                        )
