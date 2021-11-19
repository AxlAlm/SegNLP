    
#basics
from typing import List, Dict, Tuple, Union
import itertools
import os
from copy import deepcopy
from tqdm.auto import tqdm
from glob import glob
import shutil
import pandas as pd

# pytorch 
import torch

#segnlp
import segnlp.utils as utils
from segnlp.training import Trainer
from segnlp.seg_model.seg_model import SegModel



class ModelEval:


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


    def __filter_and_setup_hp_configs(self, identifed_hps: List[Tuple[str, dict]], model:str) -> List[Tuple[str, dict]]:
        """
        check which hyperparamaters ids already exist and remove them  
        """

        # get all configs
        uid2config = {c["uid"]:c for c in self.hp_configs(model)}

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
                        "path_to_models": os.path.join(self._path_to_models, model, str(hp_id)),
                        "path_to_logs": os.path.join(self._path_to_logs, model, str(hp_id)),
                        "done": False
                        }
                hp_id += 1

            hp_configs.append(config)

        return hp_configs
   

    def train(self,
                model : SegModel,
                hyperparamaters : dict,
                monitor_metric : str = "val_f1",
                gpus : Union[list, int] = None,
                overfit_batches_k : int = None
                ) -> None:

        self.training = True

        hp_sets = self.__get_hyperparam_sets(hyperparamaters)
        uid_hps_set = self.__get_hyperparamaters_uid(hp_sets)
        hp_configs = self.__filter_and_setup_hp_configs(uid_hps_set, model.name())


        device  = "cpu"
        if gpus:      
            gpu = gpus[0] if isinstance(gpus,list) else gpus
            device =  f"cuda:{gpu}"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        
        for hp_config in tqdm(hp_configs, desc="Hyperparamaters",  position=0, leave = False):    
            
            #make a folder in models for the model
            os.makedirs(hp_config["path_to_models"], exist_ok=True)

            # make a folder for model hps
            model_hp_dir = os.path.join(self._path_to_hps, model.name())
            path_to_model_hps = os.path.join(model_hp_dir, f'{hp_config["id"]}.json')
            os.makedirs(model_hp_dir, exist_ok=True)


            random_seeds = hp_config["random_seeds_todo"].copy()
            for random_seed in tqdm(random_seeds, desc = "Random Seeds", position=1, leave = False):
                
                hyperparamaters = hp_config["hyperparamaters"]

                # init model
                m = model(
                            hyperparamaters  = hyperparamaters,
                            task_dims = self.dataset.task_dims,
                            seg_task = self.dataset.seg_task
                            )

                # init folders
                os.makedirs(hp_config["path_to_logs"], exist_ok = True)
                os.makedirs(hp_config["path_to_models"], exist_ok = True)
            

                # init trainer
                trainer = Trainer(  
                                name = str(random_seed),
                                model = m,
                                dataset = self.dataset,
                                metric_fn = self.metric_fn,
                                monitor_metric = monitor_metric, 
                                optimizer_config = hyperparamaters["general"]["optimizer"], 
                                max_epochs = hyperparamaters["general"]["max_epochs"],
                                batch_size = hyperparamaters["general"]["batch_size"],
                                patience = hyperparamaters["general"].get("patience", None),
                                gradient_clip_val = hyperparamaters["general"].get("gradient_clip_val", None),
                                lr_scheduler_config = hyperparamaters["general"].get("lr_scheduler", None),
                                ground_truth_sampling_k = hyperparamaters["general"].get("ground_truth_sampling_k", None),
                                pretrain_segmenation_k = hyperparamaters["general"].get("pretrain_segmentation_k", None),
                                overfit_batches_k = overfit_batches_k,
                                path_to_models  = hp_config["path_to_models"],
                                path_to_logs = hp_config["path_to_logs"],
                                device = device
                                )

                try:
                    #run trainer
                    trainer.fit()

                    #update our config
                    hp_config["random_seeds_todo"].remove(random_seed)
                    hp_config["random_seeds_done"].append(random_seed)

                    # check if all random_seeds are done
                    if len(hp_config["random_seeds_todo"]) == 0:
                        hp_config["done"]

                    # write hyperparamters along with timestamps to a json
                    utils.save_json(hp_config, path_to_model_hps)

                except BaseException as e:
                    
                    files_to_remove = glob(os.path.join(hp_config["path_to_models"], str(random_seed) + "*.ckpt"))
                    files_to_remove += glob(os.path.join(hp_config["path_to_logs"], str(random_seed), "/*.log"))

                    for fp in files_to_remove:
                        os.remove(fp)

                    raise e


        print(" _______________  Val Scores  _______________")
        print(pd.DataFrame(self._load_logs(model.name(), split = "val")["0"]).T)





    def test(self, 
            model : SegModel,
            monitor_metric : str, 
            batch_size : int = 32,
            gpus : Union[list, str] = None
            ) -> None:


        device  = "cpu"
        if gpus:      
            gpu = gpus[0] if isinstance(gpus,list) else gpus
            device =  f"cuda:{gpu}"
        
        # load hyperparamaters
        hp_config = utils.load_json(os.path.join(self._path_to_hps, model.name(), "0.json"))

        hyperparamaters = hp_config["hyperparamaters"]
        random_seeds = hp_config["random_seeds_done"]
        path_to_models =  hp_config["path_to_models"].rsplit("/",1)[0]

        #best_pred_dfs = None
        #top_score = 0
        for random_seed in tqdm(random_seeds, desc= "random_seeds"):

            #model path 
            model_path = glob(os.path.join(path_to_models, "0", f"{random_seed}*"))[0]
            
            # init model
            m = model(
                            hyperparamaters  = hyperparamaters,
                            task_dims = self.dataset.task_dims,
                            seg_task = self.dataset.seg_task
                            )
        
            #load model weights
            m.load_state_dict(torch.load(model_path))

            # init trainer
            trainer = Trainer(  
                            name = str(random_seed),
                            model = m,
                            dataset = self.dataset,
                            metric_fn = self.metric_fn,
                            monitor_metric = monitor_metric, 
                            optimizer_config = hyperparamaters["general"]["optimizer"], 
                            max_epochs = hyperparamaters["general"]["max_epochs"],
                            batch_size = hyperparamaters["general"]["batch_size"],
                            patience = hyperparamaters["general"].get("patience", None),
                            gradient_clip_val = hyperparamaters["general"].get("gradient_clip_val", None),
                            lr_scheduler_config = hyperparamaters["general"].get("lr_scheduler", None),
                            ground_truth_sampling_k = hyperparamaters["general"].get("ground_truth_sampling_k", None),
                            pretrain_segmenation_k = hyperparamaters["general"].get("pretrain_segmentation_k", None),
                            path_to_models  = hp_config["path_to_models"],
                            path_to_logs = hp_config["path_to_logs"],
                            device = device
                            )


            # test
            trainer.test()

        print(" _______________  Test Scores  _______________")
        print(pd.DataFrame(self._load_logs(model.name(), split = "test")["0"]).T)





        # #self.rank_test(monitor_metric)

        # print(" _______________  Mean Scores  _______________")
        # mean_df = pd.DataFrame(self._load_logs()[self.best_hp]["test"]).mean().to_frame()
        # filter_list = set(["epoch", "hp_id", "random_seed", "cv", "use_target_segs", "freeze_segment_module"])
        # index = [c for c in mean_df.index if c not in filter_list]
        # mean_df = mean_df.loc[index]

        # #for baseline in self._baselines:
        # #    mean_df[baseline] = pd.DataFrame(self.baseline_scores()["test"][baseline]).mean()
            
        # print(mean_df)
