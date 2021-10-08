    
# basics
from typing import Callable
import pandas as pd
import os

from pandas.core.frame import DataFrame


# segnlp
from segnlp import utils
from segnlp import metrics
from segnlp.utils.baselines import MajorityBaseline
from segnlp.utils.baselines import RandomBaseline
from segnlp.utils.baselines import SentenceMajorityBaseline
from segnlp.utils.baselines import SentenceRandomBaseline


class Baseline:


    def __run_baseline(
                        self,
                        baseline, 
                        name : str, 
                        df:pd.DataFrame,
                        kwargs:dict,
                        metric_f: Callable,
                        task_labels: dict,
                        ):

        all_metrics  = []
        for rs in utils.random_ints(self.n_random_seeds):

            kwargs["random_seed"] = rs
            
            #init the baseline model
            bl = baseline(**kwargs)

            # run baseline
            pred_df = bl(df.copy(deep=True))

            #evaluate baseline
            metrics = metric_f(
                                pred_df = pred_df, 
                                target_df = df,
                                task_labels = task_labels
                            )

            metrics["random_seed"] = rs
            metrics["baseline"] = name
            all_metrics.append(metrics)


        score_df = pd.DataFrame(all_metrics)
        return score_df


    def __get_task_labels_ids(self) -> dict:
        task_labels_ids = {k:list(v.values()) for k,v in self.label_encoder.label2id.items()}

        for task in self.all_tasks:
            if task in task_labels_ids:
                continue
            task_labels_ids[task] = []

        task_labels_ids  = {k:v for k,v in task_labels_ids.items() if ("+" not in k  and  "seg" not in k)}

        return task_labels_ids
    

    def __get_majority_labels(self, task_labels_ids):

        if self.dataset_name == "PE":
            majority_labels = {}

            if "label" in task_labels_ids:
                majority_labels["label"] = self.label_encoder.label2id["label"]["Premise"]
            
            if "link_label" in task_labels_ids:
                majority_labels["link_label"] = self.label_encoder.label2id["link_label"]["support"]

            if "link" in task_labels_ids:
                majority_labels["link"] = None

        if self.dataset_name == "MTC":
            majority_labels = None

        return majority_labels


    def __load_data(self):
        df = pd.read_csv(self._path_to_df, index_col = 0)
        splits = utils.load_pickle_data(self._path_to_splits)

        val_df = df.loc[splits[0]["val"]]
        test_df = df.loc[splits[0]["test"]]
        return val_df, test_df
    

    def __get_sentence_baselines_scores(self,
                                        df: pd.DataFrame,
                                        metric_f: Callable,
                                        task_labels_ids : dict,
                                        majority_labels : dict
                                    ):

        score_dfs = []
        baselines = zip(["majority", "random"],[SentenceMajorityBaseline, SentenceRandomBaseline])
        for name, baseline in baselines:

            score_df = self.__run_baseline(
                            baseline = baseline,
                            name = name,
                            df = df,
                            kwargs = dict(
                                            task_labels = majority_labels if name == "majority" else task_labels_ids, 
                                            p = 1.0
                                        ),
                            metric_f = metric_f,
                            task_labels = self.task_labels
                        )
            
            score_dfs.append(score_df)

        return score_dfs


    def __get__baseline_scores(self,
                                df: pd.DataFrame,
                                metric_f: Callable,
                                task_labels_ids : dict,
                                majority_labels : dict
                                ):

        score_dfs = []
        
        val_df = df.groupby("seg_id", sort = False).first()
        baselines = zip(["majority", "random"],[MajorityBaseline, RandomBaseline])
        for name, baseline in baselines:

            score_df = self.__run_baseline(
                            baseline = baseline,
                            name = name,
                            df = val_df,
                            kwargs = dict(
                                            task_labels = majority_labels if name == "majority" else task_labels_ids, 
                                        ),
                            metric_f = metric_f,
                            task_labels = self.task_labels
                        )

            score_dfs.append(score_df)

        return score_dfs


    def baseline_scores(self):

        if os.path.exists(self._path_to_bs_scores):
            return utils.load_json(self._path_to_bs_scores)
        
        val_df, test_df = self.__load_data()   
        task_labels_ids = self.__get_task_labels_ids()
        majority_labels = self.__get_majority_labels(task_labels_ids)
        metric_f = getattr(metrics, self.metric)


        if "seg" in self.all_tasks:
            score_f = self.__get_sentence_baselines_scores
        else:
            score_f = self.__get__baseline_scores


        baseline_scores = {}
        for split, split_df in zip(["val", "test"], [val_df, test_df]):
            score_dfs = score_f(
                                df = split_df,
                                metric_f = metric_f,
                                task_labels_ids = task_labels_ids,
                                majority_labels = majority_labels,
                                )


            df = pd.concat(score_dfs)
            df.set_index("baseline", inplace = True)

            baseline_scores[split] = {
                                        "random" : df.loc["random"].to_dict("list"),
                                        "majority" : df.loc["majority"].to_dict("list")
                                    }
            
        utils.save_json(baseline_scores, self._path_to_bs_scores)
        return baseline_scores
