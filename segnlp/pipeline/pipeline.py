
#basics
import sys
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
from scipy import stats

#pytorch Lightning
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers import CometLogger


#deepsig
from deepsig import aso

#pytorch
import torch

#segnlp
from segnlp.datasets.base import DataSet
from segnlp.preprocessing import Preprocessor
from segnlp.preprocessing.dataset_preprocessor import PreProcessedDataset
from segnlp.ptl import get_ptl_trainer_args
from segnlp.ptl import PTLBase
from segnlp import get_logger
from segnlp.utils import set_random_seed
from segnlp.utils import get_time
from segnlp.utils import create_uid
from segnlp.utils import random_ints
from segnlp.utils import ensure_numpy
from segnlp.evaluation_methods import get_evaluation_method
from segnlp.features import get_feature
from segnlp.nn.utils import ModelOutput
from segnlp.visuals.hp_tune_progress import HpProgress
from segnlp.metrics import base_metric

logger = get_logger("PIPELINE")
user_dir = pwd.getpwuid(os.getuid()).pw_dir



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
                root_dir:str =f"{user_dir}/.segnlp/" #".segnlp/pipelines"       
                ):
        
        self.project = project
        self.evaluation_method = evaluation_method
        self.model = model
        self.exp_logger = None
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
        self._path_to_top_models = os.path.join(self._path_to_models, "top")
        self._path_to_tmp_models = os.path.join(self._path_to_models, "tmp")
        self._path_to_model_info = os.path.join(self._path_to_models, "model_info.json")
        self._path_to_hp_hist = os.path.join(self._path_to_models, "hp_hist.json")
        self._path_to_test_score = os.path.join(self._path_to_models, "test_scores.json")


        self.preprocessor = Preprocessor(                
                                        prediction_level=dataset.prediction_level,
                                        sample_level=dataset.sample_level, 
                                        input_level=dataset.level,
                                        features=features,
                                        encodings=encodings,
                                        other_levels=other_levels
                                        )
        self.dataset = self.process_dataset(dataset)

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

        # small hack to perserve
        self.__pp_feature_params = {f.name:f.params for f in features}
        self.__pp_encoders = self.preprocessor.encoders
        del self.preprocessor
        self.__pp_status = "inactive"

        self.__hp_tuning = False
        self.__eval_set = False


    @classmethod
    def load(self, model_dir_path:str=None, pipeline_folder:str=None, root_dir:str =f"{user_dir}/.segnlp/pipelines"):
        
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
                return self.preprocessor.process_dataset(dataset, evaluation_method=self.evaluation_method, dump_dir=self._path_to_data)
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
                        tasks=self.config["tasks"],
                        all_tasks=self.config["all_tasks"],
                        label_encoders=self.__pp_encoders,
                        prediction_level=self.config["prediction_level"],
                        task_dims={t:len(l) for t,l in self.config["task2labels"].items() if t in self.config["tasks"]},
                        feature_dims=self.config["feature2dim"],
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
        self._model.inference = True
        self.preprocessor.deactivate_labeling()
        self.__eval_set = True


    def __stat_sig(self, a_dist:List, b_dist:List, ss_test="aso"):
        """
        Tests if there is a significant difference between two distributions. Normal distribtion not needed.
        Two tests are supported. We prefer 1) (see https://www.aclweb.org/anthology/P19-1266.pdf)

        :

            1) Almost Stochastic Order

                Null-hypothesis:
                    H0 : aso-value >= 0.5
                
                i.e. ASO is not a p-value and instead the threshold is different. We want our score to be
                below 0.5, the lower it is the more sure we can be that A is better than B.    


            2) Mann-Whitney U 

                Null-hypothesis:

                    H0: P is not significantly different from 0.5
                    HA: P is significantly different from 0.5
                
                p-value >= .05


        1) is prefered

        """
        
        is_sig = False
        if ss_test == "aso":
            v = aso(a_dist, b_dist)
            is_sig = v <= 0.5

        elif ss_test == "mwu":
            v = stats.mannwhitneyu(a_dist, b_dist, alternative='two-sided')
            is_sig = v <= 0.05

        else:
            raise RuntimeError(f"'{ss_test}' is not a supported statistical significance test. Choose between ['aso', 'mwu']")

        return is_sig, v


    def __model_comparison(self, a_dist:List, b_dist:List, ss_test="aso"):
        """

        This function compares two approaches --lets call these A and B-- by comparing their score
        distributions over n number of seeds.

        first we need to figure out the proability that A will produce a higher scoring model than B. Lets call this P.
        If P is higher than 0.5 we cna say that A is better than B, BUT only if P is significantly different from 0.5. 
        To figure out if P is significantly different from 0.5 we apply a significance test.

        https://www.aclweb.org/anthology/P19-1266.pdf
        https://export.arxiv.org/pdf/1803.09578

        """

        a_dist = ensure_numpy(a_dist)
        b_dist = ensure_numpy(b_dist)

        if all(np.sort(a_dist) == np.sort(b_dist)):
            return False, 0.0, 1.0

        larger_than = a_dist >= b_dist
        x = larger_than == True
        prob = sum(larger_than == True) / larger_than.shape[0]

        a_better_than_b = None
        v = None
        if prob > 0.5:
            
            is_sig, v = self.__stat_sig(a_dist, b_dist, ss_test=ss_test)

            if is_sig:
                a_better_than_b = True

        return a_better_than_b, prob, v

    
    def train(self,
                    hyperparamaters:dict,
                    ptl_trn_args:dict={},
                    n_random_seeds:int=6,
                    random_seed:int=None,
                    save_choice:str="best",
                    monitor_metric:str = "val_f1",
                    ss_test:str="aso",
                    debug:bool=False,
                    override:bool=False
                    ):

        # if ptl_trn_args.get("gradient_clip_val", 0.0) != 0.0:
        #     hyperparamaters["gradient_clip_val"] = ptl_trn_args["gradient_clip_val"]
        
        self.__hp_tuning = True
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
    
        create_hp_uid = lambda x: create_uid("".join(list(map(str, x.keys()))+ list(map(str, x.values()))))
        hp_dicts = {create_hp_uid(hp):{"hyperparamaters":hp} for hp in set_hyperparamaters}
        hp_dicts.update(hp_hist)

        hpp = HpProgress(
                        keys=keys,
                        hyperparamaters=hp_dicts,
                        n_seeds = n_random_seeds,
                        best=best_model_info.get("uid", None),
                        show_n=3,
                        debug=debug,
                        )

        for hp_uid, hp_dict in hp_dicts.items():    
            
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
                random_seeds = random_ints(n_random_seeds)

            for ri, random_seed in enumerate(random_seeds, start=1):
                hpp.refresh(hp_uid, 
                            progress = ri, 
                            best_score = best_model_score if best_model_score  else "-",
                            mean_score = np.mean(model_scores) if model_scores else "-"
                            )

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
                is_better, p, v = self.__model_comparison(model_scores, best_scores, ss_test=ss_test)
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

                hpp.set_top(hp_uid)

            if os.path.exists(self._path_to_tmp_models):
                shutil.rmtree(self._path_to_tmp_models)

            hpp.update()
        
        hpp.close()
            
        with open(self._path_to_hp_hist, "w") as f:
            json.dump(hp_dicts, f, indent=4)

        with open(self._path_to_model_info, "w") as f:
            json.dump(best_model_info, f, indent=4)

        return best_model_info


    def _fit(    self,
                hyperparamaters:dict,
                ptl_trn_args:dict={},
                save_choice:str = "best",  
                random_seed:int = 42,
                monitor_metric:str = "val_f1",
                model_id:str=None
                ):


        if model_id is None:
            model_id = create_uid("".join(list(map(str, hyperparamaters.keys())) + list(map(str, hyperparamaters.values()))))

        set_random_seed(random_seed)

        hyperparamaters["random_seed"] = random_seed
        self.dataset.batch_size = hyperparamaters["batch_size"]
        hyperparamaters["monitor_metric"] = monitor_metric


        model = deepcopy(self.model)
    
        if self.exp_logger:
            ptl_trn_args["logger"] = self.exp_logger
        else:
            ptl_trn_args["logger"] = None
        
        mid_folder = "top" if not self.__hp_tuning else "tmp"
        exp_model_path = os.path.join(self._path_to_models, "tmp", model_id, str(random_seed))
        
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


        if self.exp_logger:
            self.exp_logger.set_id(model_id)
            self.exp_logger.log_hyperparams(hyperparamaters)

            if isinstance(exp_logger, CometLogger):
                self.exp_logger.experiment.add_tags([self.project, self.id])
                self.exp_logger.experiment.log_others(exp_config)


        ptl_trn_args = get_ptl_trainer_args( 
                                        ptl_trn_args=ptl_trn_args,
                                        hyperparamaters=hyperparamaters, 
                                        exp_model_path=exp_model_path,
                                        save_choice=save_choice, 
                                        #prefix=model_id,
                                        )

        model_fp, model_score = get_evaluation_method(self.evaluation_method)(
                                                                                model_args = model_args,
                                                                                ptl_trn_args = ptl_trn_args,
                                                                                dataset = self.dataset,
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
 

    def test(   self, 
                model_folder:str=None,
                ptl_trn_args:dict={},
                monitor_metric:str = "val_f1",
                seg_preds:str=None,
                ):

        self.dataset.split_id = 0


        with open(self._path_to_model_info, "r") as f:
            model_info = json.load(f)

        models_to_test =  model_info["outputs"]
        best_seed = model_info["best_model"]["random_seed"]
        

        best_model_scores = None
        best_model_outputs = None
        seed_scores = []
        seeds = []

        for seed_model in models_to_test:

            seeds.append(seed_model["random_seed"])

            with open(seed_model["config_path"], "r") as f:
                model_config = json.load(f)

            hyperparamaters = model_config["args"]["hyperparamaters"]
            self.dataset.batch_size = hyperparamaters["batch_size"]

            trainer = setup_ptl_trainer( 
                                        ptl_trn_args=ptl_trn_args,
                                        hyperparamaters=hyperparamaters, 
                                        exp_model_path=None,
                                        save_choice=None, 
                                        )

            model_config["args"]["model"] = deepcopy(self.model)
            model_config["args"]["label_encoders"] = self.__pp_encoders
            model = PTLBase.load_from_checkpoint(seed_model["path"], **model_config["args"])
            scores = trainer.test(
                                    model=model, 
                                    test_dataloaders=self.dataset.test_dataloader(),
                                    verbose=0
                                    )

            test_output = pd.DataFrame(model.outputs["test"])


            if seg_preds is not None:
                test_output["seg"] = "O"

                #first we get all the token rows
                seg_preds = seg_preds[seg_preds["token_id"].isin(test_output["token_id"])]

                # then we sort the seg_preds
                seg_preds.index = seg_preds["token_id"]
                seg_preds = seg_preds.reindex(test_output["token_id"])

                assert np.array_equal(seg_preds.index.to_numpy(), test_output["token_id"].to_numpy())
                
                #print(seg_preds["seg"])
                test_output["seg"] = seg_preds["seg"].to_numpy()
                seg_mask = test_output["seg"] == "O"

                task_scores = []
                for task in self.config["subtasks"]:
                    default_none =  "None" if task != "link" else 0
                    test_output.loc[seg_mask, task] = default_none
                    task_scores.append(base_metric(
                                                    targets=test_output[f"T-{task}"].to_numpy(), 
                                                    preds=test_output[task].to_numpy(), 
                                                    task=task, 
                                                    labels=self.config["task2labels"][task]
                                                    ))

                scores = [pd.DataFrame(task_scores).mean().to_dict()]
              


            if seed_model["random_seed"] == best_seed:
                best_model_scores = pd.DataFrame(scores)
                best_model_outputs = pd.DataFrame(test_output)

            seed_scores.append(scores[0])
    
        df = pd.DataFrame(seed_scores, index=seeds)
        mean = df.mean(axis=0)
        std = df.std(axis=0)

        final_df = df.T
        final_df["mean"] = mean
        final_df["std"] = std
        final_df["best"] = best_model_scores.T
        
        with open(self._path_to_test_score, "w") as f:
            json.dump(seed_scores, f, indent=4)
        
        print(final_df)
        return final_df, best_model_outputs


    def predict(self, doc:str):

        if not self.__eval_set:
            raise RuntimeError("Need to set pipeline to evaluation mode by using .eval() command")

        Input = self.preprocessor([doc])
        Input.sort()
        Input.to_tensor(device="cpu")

        output = self._model(Input)
        return output
        

