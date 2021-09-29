    
#basics
from typing import List, Dict, Tuple, Union
import itertools
import json
import os
from copy import deepcopy
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
        create_hp_id = lambda x: utils.create_uid("".join(list(map(str, x.keys())) + list(map(str, x.values()))))
        hp_ids = [create_hp_id(hp) for hp in hp_sets]


        # add ids
        hp_dicts = dict(zip(hp_ids,hp_sets))
        
        # hp_ids of the saved hyperparamaters
        done_hp_ids = set([fp.replace(".json", "") for fp in os.listdir(self._path_to_hps)])

        for hp_id in list(hp_dicts.keys()):
            if hp_id in done_hp_ids:
                logger.info(f"WARNING! Hyperparamater with id {hp_id} has already been run. Will remove from hyperparamter tuning.")
                hp_dicts.pop(hp_id)

    
        return hp_dicts


    def __dump_hps( self,
                    hp_id:str,
                    hyperparamaters: dict, 
                    random_seeds: list,
                    ):

        hyperparamaters = deepcopy(hyperparamaters)

        # #fix objects
        # for group, hps in hyperparamaters.items():
        #     for k,v in hps.items():

        #         if isinstance(v, dict):
        #             pass

        #         elif not isinstance(v, (str, int, float)):
        #             hyperparamaters[group][k] = str(hyperparamaters[group][k])
                
        #         else:
        #             hyperparamaters[group][k]
        
   
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



    def train(self,
                hyperparamaters:dict,
                n_random_seeds:int=6,
                random_seed:int=None,
                monitor_metric:str = "val_f1",
                gpus : list = [],
                overfit_n_batches : int = None
                ):

        self.training = True

        hp_dicts = self.__create_hyperparam_sets(hyperparamaters)


        if not gpus:
            device = "cpu"
        else:
            device =  f"cuda:{gpus[0]}"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        
        for hp_id, hps in tqdm(hp_dicts.items(), desc="Hyperparamaters",  position=0):    
            
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
                            hyperparamaters = hps,
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
