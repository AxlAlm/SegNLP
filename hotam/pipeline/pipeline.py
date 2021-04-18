
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
        self._model.inference = True
        self.preprocessor.deactivate_labeling()
        self.__eval_set = True


    # def ss_test(self):

    #     """

    #     Statistical testing is done according to Evaluation 4 mentioned in the paper below

    #     https://arxiv.org/pdf/1803.09578.pdf


    #     Essentially, we test if some model A is more likely to produce a better working model than B.

    #     A model is a NN architecture with some hyperparamater.
    #     For example, A and B can both be LSTM_CRF but with different hyperparamaters.

    #     Approach is "formalizes" as following:

    #          P( 
    #             Ψ(Test)A(Train,Dev,Rnd) 
    #             ≥
    #             Ψ(Test)B(Train,Dev,Rnd)
    #             ) > 0.5

    #     A model is trained on a train set and evaluated on a Dev(val) set over a sequence of random_seeds:

    #         score_distribution = []
    #         for seed in sequence_of_random_seeds:
    #             score_distribution.append(fit(model, train, dev))


    #     Mann-Whitney U test
    #     Wilcoxon signed-rank test



    #     Returns
    #     -------
    #     [type]
    #         [description]
    #     """

    #     if "random_seed" not in hyperparamaters:
    #         hyperparamaters["random_seed"] = 42

    #     seed_scores = []
    #     for seed in random_seeds:
            
    #         self.random_seed = seed
    #         set_random_seed(seed)
    #         hyperparamaters["random_seed"] = seed

    #         hp_tune_results =  self.tune_hps(
    #                                     hyperparamaters = hyperparamaters,
    #                                     ptl_trn_args = ptl_trn_args,
    #                                     save_choice =  save_choice                                           
    #                                     )
            
    #         seed_scores.append(
    #                             self.tune_hps(
    #                                             hyperparamaters = hyperparamaters,
    #                                             ptl_trn_args = ptl_trn_args,
    #                                             save_choice =  save_choice                                           
    #                                             )
    #                                             )

    #     # DO SOME STATISTICAL SIG TESTING
    #     return seed_scores



    def __model_comparison(self, a_dist:List, b_dist:List, ss_test="aso"):
        """

        This function compares two approaches --lets call these A and B-- by comparing their score
        distributions over n number of seeds.


        first we need to figure out the proability that A will produce a higher scoring model than B. Lets call this P.
        If P is higher than 0.5 we cna say that A is better than B, BUT only if P is significantly different from 0.5. 
        To figure out if P is significantly different form 0.5 we apply a significance test. We have chosen to use:

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

        https://www.aclweb.org/anthology/P19-1266.pdf
        https://export.arxiv.org/pdf/1803.09578


        """
        larger_than = a_dist >= a_dist
        prob = sum(larger_than == True) / larger_than.shape[0]

        a_better_than_b = None
        if prob > 0.5:
            
            if ss_test == "aso":
                aso(df["one"].to_numpy(),df["two"].to_numpy())
                a_better_than_b = v <= 0.5

            elif ss_test == "mwu":
                v = stats.mannwhitneyu(df["one"].to_numpy(), df["two"].to_numpy(), alternative='two-sided')
                a_better_than_b = v <= 0.05
        
        return a_better_than_b

    
    def select_model(   self,
                    hyperparamaters:dict,
                    ptl_trn_args:dict={},
                    random_seed=42,
                    n_random_seeds:int=None,
                    save_choice:str="last",
                    monitor_metric:str = "val_f1",
                    ss_test:str="aso"
                    ):


        if n_random_seeds is not None:
            random_seeds = np.random.randint(10**6,size=(n_random_seed,))
        else:
            random_seeds = [random_seed]

        set_hyperparamaters = self.__create_hyperparam_sets(hyperparamaters)

        best_hp_df = None
        for hyperparamaters in set_hyperparamaters:
            
            model_scores = []
            for seed in random_seeds:
                model_scores.append(self.fit(
                                                hyperparamaters=hyperparamaters,
                                                ptl_trn_args = ptl_trn_args,
                                                save_choice=save_choice,
                                                random_seed=seed,
                                                monitor_metric=monitor_metric
                                                ))

            model_score_df = pd.concat(model_scores)

            if best_hp_df is not None:
                a_dist = model_score_df["two"]
                b_dist = best_hp_df["f1"]

                if self.__model_comparison(a_dist, b_dist, test=ss_test):
                    best_hp_df = model_score_df         
            else:
                best_hp_df = model_score_df

            hp_scores.append(model_scores)

        #model_ranking = pd.DataFrame(model_scores)
        #model_ranking.sort_values("score", ascending=False if "loss" in monitor_metric else True, inplace=True)
        
        # with open(os.path.join(self._path_to_models,"model_rankings.json"), "w") as f:
        #     json.dump(model_ranking.to_dict(), f, indent=4)

        return model_scores



    def run(    self,
                hyperparamaters:dict,
                ptl_trn_args:dict={},
                exp_logger:LightningLoggerBase=None,


                )

        self.exp_logger = exp_logger

        pass


    def fit(    self,
                hyperparamaters:dict,
                ptl_trn_args:dict={},
                save_choice:str = "last",  
                random_seed:int = 42,
                monitor_metric:str = "val_f1",
                ):

        hyperparamaters["random_seed"] = random_seed
        self.dataset.batch_size = hyperparamaters["batch_size"]

        model = deepcopy(self.model)
    
        if self.exp_logger:
            ptl_trn_args["logger"] = self.exp_logger
        else:
            ptl_trn_args["logger"] = None

        model_unique_str = "".join(
                                        [model.name()]
                                        + list(map(str, hyperparamaters.keys()))
                                        + list(map(str, hyperparamaters.values()))
                                    )
        model_id = create_uid(model_unique_str)
        exp_model_path = os.path.join(self._path_to_models, random_seed, model_id)

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


        return {
                "model_id":model_id, 
                "score":model_score, 
                "monitor_metric":monitor_metric,
                "path":model_fp, 
                "config_path": os.path.join(exp_model_path, "model_config.json")
                }
 



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
        model = PTLBase.load_from_checkpoint(ckpt_fp, **model_config["args"])
        scores = trainer.test(
                                model=model, 
                                test_dataloaders=self.dataset.test_dataloader()
                                )
        
        output_df = pd.DataFrame(model.outputs["test"])
        output_df["text"] = output_df["text"].apply(np.vectorize(lambda x:x.decode("utf-8")))

        print(output_df.head(10))
        return scores, output_df
        

    def predict(self, doc:str):

        if not self.__eval_set:
            raise RuntimeError("Need to set pipeline to evaluation mode by using .eval() command")

        Input = self.preprocessor([doc])
        Input.sort()
        Input.to_tensor(device="cpu")

        output = self._model(Input)
        return output
        
