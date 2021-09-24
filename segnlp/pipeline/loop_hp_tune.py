    
#basics
from typing import List, Dict, Tuple, Union
import itertools
import json
import os
from copy import deepcopy
from tqdm.auto import tqdm


#segnlp
from segnlp import get_logger
import segnlp.utils as utils


logger = get_logger("LOOP HP TUNING")


class HPTuneLoop:


    def __create_hyperparam_sets(self, hyperparamaters:dict) -> Dict[str, dict]:
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
        create_hp_uid = lambda x: utils.create_uid("".join(list(map(str, x.keys())) + list(map(str, x.values()))))
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


    def __dump_hp_uid(self, hp_uid: str) -> None:

        with open(self._path_to_hp_hist, "a") as f:
            f.write(hp_uid + "\n")


    def __dump_hps( self,
                    hp_uid:str,
                    hyperparamaters: dict, 
                    ):

        hyperparamaters = deepcopy(hyperparamaters)

        #fix objects
        for t, hps in hyperparamaters.items():
                for k,v in hps.items():
                    if not isinstance(v, (str, int, float)):
                        hyperparamaters[t][k] = str(hyperparamaters[t][k])
            
   
        time = utils.get_time()
        config = {
                    "hp_uid": hp_uid,
                    "time": str(time),
                    "timestamp": str(time.timestamp()),
                    "hyperparamaters": hyperparamaters,
                    }

        #### check if the hp_uids are already tested and logged
        with open(self._path_to_hp_json, "a") as f:
            f.write(json.dumps(config, indent=4) + "\n")



    def train(self,
                hyperparamaters:dict,
                n_random_seeds:int=6,
                random_seed:int=None,
                monitor_metric:str = "val_f1",
                ):

        self.training = True

        hp_dicts = self.__create_hyperparam_sets(hyperparamaters)
        
        for hp_uid, hps in tqdm(hp_dicts.items(), desc="Hyperparamaters",  position=0):    
        
            if random_seed is not None and isinstance(random_seed, int):
                random_seeds = [random_seed]
            else:
                random_seeds = utils.random_ints(n_random_seeds)

            for random_seed in tqdm(random_seeds, desc = "Random Seeds", position=1):
                
                utils.set_random_seed(random_seed)

                self.fit(
                        model_id = f"{hp_uid}-{random_seed}",
                        hyperparamaters = hps,
                        monitor_metric = monitor_metric,
                        )

            # write hp uid to a .txt file
            self.__dump_hp_uid(hp_uid)

            # write hyperparamters along with timestamps to a json
            self.__dump_hps(hp_uid, hps)
