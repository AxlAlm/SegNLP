
# basics
import numpy as np
import pandas as pd
import os

# segnlp
from segnlp.utils.stat_sig import compare_dists


class RankItem(dict):
    
    def __init__(self, id:str, v:float, scores:np.ndarray, metric:str):
        self["id"] = id
        self["v"] = v
        self["max"] = np.max(scores)
        self["min"] = np.min(scores)
        self["mean"] = np.mean(scores)
        self["std"] = np.std(scores)
        self["metric"] = metric


class Ranking:

    @property
    def rankings(self):
        return pd.read_csv(self._path_to_rankings, index_col=0)

    @property
    def best_hp(self):
        return self.rankings.iloc[0]["id"]


    def __get_score_dists(self, monitor_metric: str, split:str):
        
        hp_score_dists = {}
        log_dfs = self._load_logs()
        for hp_id in self.hp_ids:

            if split not in log_dfs[hp_id]:
                continue

            df = log_dfs[hp_id][split]
            max_df = df.groupby("random_seed")[monitor_metric].max()
            hp_score_dists[hp_id] = {
                                    "random_seed": max_df.index.to_numpy(),
                                    monitor_metric: max_df.to_numpy()
                                    }

        # get baslines scores 
        baseline_scores = self.baseline_scores()[split]
 
        # create a dict with score dists
        score_dists = {**hp_score_dists, **baseline_scores}

        return score_dists


    def __set_baseline_ranks(self, score_dists:dict, monitor_metric:str):

        a = score_dists["majority"][monitor_metric]
        b = score_dists["random"][monitor_metric]

        # we see if majority is better than random
        majority_is_best, m_v = compare_dists(a, b)
        if majority_is_best:
            return [
                    RankItem(id = "majority", v = m_v, scores = a, metric = monitor_metric),
                    RankItem(id = "random", v = -1, scores = b, metric = monitor_metric),
                    ]
                    
        # we see if random is better than majority
        random_is_best, r_v = compare_dists(b, a)
        if random_is_best:
            return [
                    RankItem(id = "random", v = r_v, scores = b, metric = monitor_metric),
                    RankItem(id = "majority", v = -1, scores = a, metric = monitor_metric),
                    ]

        return [
                    RankItem(id = "random", v = -1, scores = b, metric = monitor_metric),
                    RankItem(id = "majority", v = -1, scores = a, metric = monitor_metric),
                    ]


    def __rank(self, hp_ids:list, rankings:list,  score_dists:dict, monitor_metric:str):

        for hp_id in hp_ids:
            a = score_dists[hp_id][monitor_metric]

            # we will look for a rank
            position = 0
            while True:
                
                # if the ranking is last
                if position >= len(rankings):
                    rankings.insert(-1, RankItem(
                                                    id = hp_id, 
                                                    v = -1, 
                                                    scores = a, 
                                                    metric = monitor_metric
                                                    )
                                        )
                    break

                id_to_comp = rankings[position]["id"]
                
                b = score_dists[id_to_comp][monitor_metric]

                a_is_better, v = compare_dists(a, b)

                if a_is_better:
                    rankings.insert(position, RankItem(
                                                        id = hp_id, 
                                                        v = v, 
                                                        scores = a,
                                                        metric = monitor_metric
                                                        ))
                    break

                position += 1
        
        return rankings

    
    def rank_hps(self, monitor_metric:str):

        print("Ranking Hyperparamaters ... ")

        hp_ids = self.hp_ids

        score_dists = self.__get_score_dists(
                                    monitor_metric = monitor_metric,
                                    split = "val"
                                )

        # if we have rankings we fetch them
        if os.path.exists(self._path_to_rankings):

            # load the old rankings
            ranking_df = pd.read_csv(self._path_to_rankings, index_col=0)
            rankings = ranking_df.to_dict("records")

            # we filter out ids we already have in the rankings from the hp_score_dist 
            done_ids = set(list(ranking_df["id"]))
            hp_ids = [i for i in hp_ids if i not in done_ids]

        else:
            # if we dont have rankings we create a new list and add random and
            #  majoirty baseline to the rankings as a start
            rankings = self.__set_baseline_ranks(score_dists, monitor_metric)

        
        # ranking will place all hps in hp_score_dist
        rankings = self.__rank(
                        hp_ids = hp_ids,
                        rankings = rankings, 
                        score_dists = score_dists,
                        monitor_metric = monitor_metric
                        )
        
        rankings = pd.DataFrame(rankings)
        print(" ___________ Current Hyperparamater Rankings ____________ ")
        print(rankings)
        rankings.to_csv(self._path_to_rankings)


    def rank_test(self, monitor_metric:str):

        # get test scores
        score_dists = self.__get_score_dists(
                                    monitor_metric = monitor_metric,
                                    split = "test"
                                )
        
        # if we dont have rankings we create a new list and add random and
        #  majoirty baseline to the rankings as a start
        rankings = self.__set_baseline_ranks(score_dists, monitor_metric)

        
         # ranking will place all hps in hp_score_dist
        rankings = self.__rank(
                        hp_ids = [self.best_hp],
                        rankings = rankings, 
                        score_dists = score_dists,
                        monitor_metric = monitor_metric
                        )

        rankings = pd.DataFrame(rankings)
        print(" _______________ Test Rankings ________________ ")
        print(rankings)

