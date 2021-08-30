    
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


class Trainer:


    def __create_hyperparam_sets(self, hyperparamaters:dict) -> List:
        """creates a set of hyperparamaters for hyperparamaters based on given hyperparamaters lists.
        takes a hyperparamaters and create a set of new paramaters given that any
        paramater values are list of values.
        """
        group_variations = []
        for group, gdict in hyperparamaters.items():
            vs = [[v] if not isinstance(v, list) else v for v in gdict.values()]
            group_sets = [dict(zip(gdict.keys(),v)) for v in itertools.product(*vs)]
            group_variations.append(group_sets)

        
        hp_sets = [dict(zip(hyperparamaters.keys(),v)) for v in itertools.product(*group_variations)]
        return hp_sets


    def __save_model_config(  self,
                            model_args:str,
                            save_choice:str, 
                            monitor_metric:str,
                            exp_model_path:str,
                            ):

        #dumping the arguments
        model_args_c = deepcopy(model_args)
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
                hyperparamaters:dict,
                ptl_trn_args:dict={},
                save_choice:str = "best",  
                random_seed:int = 42,
                monitor_metric:str = "val_f1",
                model_id:str=None,
                ):

        if model_id is None:
            model_id = utils.create_uid("".join(list(map(str, hyperparamaters.keys())) + list(map(str, hyperparamaters.values()))))

        utils.set_random_seed(random_seed)

        hyperparamaters["general"]["random_seed"] = random_seed
        hyperparamaters["general"]["monitor_metric"] = monitor_metric

        #loading our preprocessed dataset
        data_module = utils.DataModule(
                                path_to_data = self._path_to_data,
                                batch_size = hyperparamaters["general"]["batch_size"],
                                )
        
        exp_model_path = os.path.join(self._path_to_models, "tmp", model_id, str(random_seed))
        
        if os.path.exists(exp_model_path):
            shutil.rmtree(exp_model_path)
            
        os.makedirs(exp_model_path, exist_ok=True) 

        model_args = dict(
                        hyperparamaters = hyperparamaters,
                        label_encoder = self.label_encoder,
                        feature_dims = self.config["feature2dim"],
                        metric = self.metric
                        )

        self.__save_model_config(
                                model_args=model_args,
                                save_choice=save_choice,
                                monitor_metric=monitor_metric,
                                exp_model_path=exp_model_path
                                )


        ptl_trn_args = get_ptl_trainer_args( 
                                        ptl_trn_args = ptl_trn_args,
                                        hyperparamaters = hyperparamaters, 
                                        exp_model_path = exp_model_path,
                                        save_choice = save_choice, 
                                        )

        model_fp, model_score = self._eval(
                                            model_args = model_args,
                                            ptl_trn_args = ptl_trn_args,
                                            data_module = data_module,
                                            save_choice=save_choice,
                                            )

        return {
                "model_id":model_id, 
                "score":model_score, 
                "monitor_metric":monitor_metric,
                "random_seed":random_seed,
                "path":model_fp, 
                "config_path": os.path.join(exp_model_path, "model_config.json")
                }
 

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

        # if ptl_trn_args.get("gradient_clip_val", 0.0) != 0.0:
        #     hyperparamaters["gradient_clip_val"] = ptl_trn_args["gradient_clip_val"]
        
        keys = list(hyperparamaters.keys())
        set_hyperparamaters = self.__create_hyperparam_sets(hyperparamaters)

        # if we have done previous tuning we will start from where we ended, i.e. 
        # from the previouos best Hyperparamaters
        if os.path.exists(self._path_to_model_info):

            with open(self._path_to_model_info, "r") as f:
                best_model_info = json.load(f)

            best_scores = best_model_info["scores"]
        else:
            best_scores = None
            best_model_info = {}

        
        if os.path.exists(self._path_to_hp_hist):
            with open(self._path_to_hp_hist, "r") as f:
                hp_hist = json.load(f)
        else:
            hp_hist = {}
    
        create_hp_uid = lambda x: utils.create_uid("".join(list(map(str, x.keys()))+ list(map(str, x.values()))))
        hp_dicts = {create_hp_uid(hp):{"hyperparamaters":hp} for hp in set_hyperparamaters}
        hp_dicts.update(hp_hist)


        print("_" * (os.get_terminal_size().columns -1))

        for hp_uid, hp_dict in tqdm(hp_dicts.items(), desc="Hyperparamaters",  position=0):    
            
            if hp_uid in hp_hist and not override:
                logger.info(f"Following hyperparamaters {hp_uid} has already been tested over n seed. Will skip this set.")
                continue

            best_model_score = None
            best_model = None
            model_scores = []
            model_outputs = []

            if random_seed is not None and isinstance(random_seed, int):
                random_seeds = [random_seed]
            else:
                random_seeds = utils.random_ints(n_random_seeds)

            for ri, random_seed in tqdm(list(enumerate(random_seeds, start=1)), desc = "Random Seeds", position=1):

                output = self._fit(
                                    hyperparamaters=hp_dict["hyperparamaters"],
                                    ptl_trn_args = ptl_trn_args,
                                    save_choice=save_choice,
                                    random_seed=random_seed,
                                    monitor_metric=monitor_metric,
                                    model_id=hp_uid,
                                    )

                score = output["score"]

                if best_model_score is None:
                    best_model_score = score
                    best_model = output
                else:
                    if score > best_model_score:
                        best_model_score = score
                        best_model = output
                
                model_outputs.append(output)
                model_scores.append(score)


            hp_dicts[hp_uid]["uid"] = hp_uid
            hp_dicts[hp_uid]["scores"] = model_scores
            hp_dicts[hp_uid]["score_mean"] = np.mean(model_scores)
            hp_dicts[hp_uid]["score_median"] = np.median(model_scores)
            hp_dicts[hp_uid]["score_max"] = np.max(model_scores)
            hp_dicts[hp_uid]["score_min"] = np.min(model_scores)
            hp_dicts[hp_uid]["monitor_metric"] = monitor_metric
            hp_dicts[hp_uid]["std"] = np.std(model_scores)
            hp_dicts[hp_uid]["ss_test"] = ss_test
            hp_dicts[hp_uid]["n_random_seeds"] = n_random_seeds
            hp_dicts[hp_uid]["hyperparamaters"] = hp_dict["hyperparamaters"]
            hp_dicts[hp_uid]["outputs"] = model_outputs
            hp_dicts[hp_uid]["best_model"] = best_model
            hp_dicts[hp_uid]["best_model_score"] = best_model_score
            hp_dicts[hp_uid]["progress"] = n_random_seeds


            if best_scores is not None:
                is_better, p, v = self.model_comparison(model_scores, best_scores, ss_test=ss_test)
                hp_dicts[hp_uid]["p"] = p
                hp_dicts[hp_uid]["v"] = v

            if best_scores is None or is_better:
                best_scores = model_scores

                hp_dicts[hp_uid]["best_model"]["path"] = hp_dicts[hp_uid]["best_model"]["path"].replace("tmp","top")
                hp_dicts[hp_uid]["best_model"]["config_path"] = hp_dicts[hp_uid]["best_model"]["config_path"].replace("tmp","top")

                updated_outputs = []
                for d in hp_dicts[hp_uid]["outputs"]:
                    d["path"] = d["path"].replace("tmp","top")
                    d["config_path"] = d["config_path"].replace("tmp","top")
                    updated_outputs.append(d)
                
                hp_dicts[hp_uid]["outputs"] = updated_outputs
                best_model_info = hp_dicts[hp_uid]

                if os.path.exists(self._path_to_top_models):
                    shutil.rmtree(self._path_to_top_models)
                    
                shutil.move(self._path_to_tmp_models, self._path_to_top_models)


            if os.path.exists(self._path_to_tmp_models):
                shutil.rmtree(self._path_to_tmp_models)

                    
            with open(self._path_to_hp_hist, "w") as f:
                json.dump(hp_dicts, f, indent=4)
    
            with open(self._path_to_model_info, "w") as f:
                json.dump(best_model_info, f, indent=4)


        return best_model_info
