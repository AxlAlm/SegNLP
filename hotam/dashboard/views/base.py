
#basics
import pandas as pd
import re
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import json
import imgkit
import base64
import os
import math


#dash
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import dash_daq as daq


#plotly
import plotly
import plotly.express as px
import plotly.graph_objs as go

#hotviz
import hotviz


#utils
from hotam.dashboard.utils import get_filter, fig_layout, get_visible_info, update_visible_info
from hotam.dashboard.views.visuals import *


th_path  = "/tmp/text_highlight.png"



class ViewBase:
    """
    Class containing all functions used for callbacks.

    To create a new view we can reuse the functions created for another view.
    """


    def update_project_dropdown(self, n):
        projects = self.db.get_projects()
        return [{"label":p, "value":p} for p in projects]


    def update_live_exp_dropdown(self, n):
        exps = sorted(self.db.get_live_exps_ids())
        return [{"label":e, "value":e} for e in exps]


    def update_done_exp_dropdown(self, n):
        exps = sorted(self.db.get_done_exps_ids())
        return [{"label":e, "value":e} for e in exps]


    def update_task_dropdown(self, project):
        tasks = sorted(self.db.get_project_tasks(project))
        return [{"label":t, "value":t} for t in tasks]


    def update_output(self, value):
        if value:
            return {'display': 'none'}
        else:
            return {'display': 'block'}
    

    def get_exp_data(self, experiment_id:str, last_epoch:int):

        filter_by = {"experiment_id":experiment_id}
    
        exp_config = self.db.get_exp_config(filter_by)

        filter_by["epoch"] =  { "$lte": last_epoch}
        scores = self.db.get_scores(filter_by)
        scores = scores.to_dict()

        filter_by["epoch"] = last_epoch
        outputs = self.db.get_outputs(filter_by).get("data", {})
        
        if "_id" in exp_config:
            exp_config.pop("_id")
        
        if "_id" in scores:
            scores.pop("_id")
        
        if "_id" in outputs:
            outputs.pop("_id")

        exp_data = {
                "exp_config": exp_config,
                "experiment_id": experiment_id,
                "epoch": last_epoch,
                "scores": scores,
                "outputs": outputs,
                }
        
        return exp_data, {"display":"block"}


    def get_data_table(self, data_cache):
        data = data_cache["exp_config"]["dataset_stats"]
        df = pd.DataFrame(data)
        return make_table(df, "Dataset Statistics")


    def get_config(self, config_value, data_cache):
        exp_config = data_cache.get("exp_config", {})

        if config_value == "exp config":
            exp_config.pop("trainer_args")
            exp_config.pop("hyperparamaters")
            exp_config.pop("dataset_config")
            exp_config.pop("_id")
            return json.dumps(exp_config, indent=4)
        else:
            return json.dumps(exp_config.get(config_value,{}), indent=4)


    def update_loss_graph(self, data_cache, fig_state):

        fig_state = go.Figure(fig_state)

        if not data_cache["scores"]:
            return fig_state, {"display":"none"}


        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]
        tasks = data_cache["exp_config"]["tasks"]
        
        task_loss = [task+"-loss" for task in tasks]

        figure = make_lineplot(data, task_loss, "Loss")

        last_vis_state = get_visible_info(fig_state)
        current_vis_state = get_visible_info(figure)
        if last_vis_state.keys() == current_vis_state.keys():
            update_visible_info(figure, last_vis_state)

        return figure, {"display":"block"}


    def update_task_metric_graph(self, data_cache, fig_state):


        fig_state = go.Figure(fig_state)

        if not data_cache["scores"]:
            return fig_state, {"display":"none"}


        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]
        
        tasks = data_cache["exp_config"]["tasks"]
        metrics = data_cache["exp_config"]["metrics"].copy()
        metrics.remove("confusion_matrix")       


        task_metrics = []
        for task in tasks:
            for metric in metrics:
                task_metrics.append("-".join([task,metric]))
    
        figure = make_lineplot(data, task_metrics, "Task Scores")


        last_vis_state = get_visible_info(fig_state)

        if data_cache["epoch"] == 0:
            last_vis_state = {k:False for k in task_metrics if "f1" not in k}

        current_vis_state = get_visible_info(figure)
        if last_vis_state.keys() == current_vis_state.keys():
            update_visible_info(figure, last_vis_state)

        return figure, {"display":"block"}


    def update_class_metric_graph(self, data_cache, fig_state):
        
        fig_state = go.Figure(fig_state)

        if not data_cache["scores"]:
            return fig_state, {"display":"none"}


        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]

        tasks = data_cache["exp_config"]["tasks"]
        metrics = data_cache["exp_config"]["metrics"].copy()
        metrics.remove("confusion_matrix")
        task2labels = data_cache["exp_config"]["dataset_config"]["task_labels"]

        filter_columns = []
        for task in tasks:
            classes = task2labels[task]
            for c in classes:
                for metric in metrics:
                    filter_columns.append("-".join([task, str(c), metric]).lower())

    
        figure =  make_lineplot(data, filter_columns, "Class Scores")

        last_vis_state = get_visible_info(fig_state)


        if data_cache["epoch"] == 0:
            last_vis_state = {k:False for k in filter_columns if "f1" not in k}

        current_vis_state = get_visible_info(figure)
        if last_vis_state.keys() == current_vis_state.keys():
            update_visible_info(figure, last_vis_state)

        return figure, {"display":"block"}


    def update_sample_id_dropdown(self, experiment_id):
        exp_config = self.db.get_exp_config({"experiment_id":experiment_id})

        if exp_config is None:
            return [], None

        ids = exp_config["dataset_config"]["tracked_sample_ids"]["0"]

        options = [{"label":str(i), "value":str(i)}  for i in ids]
        value = str(ids[-1])

        return options, value


    def update_highlight_text(self, sample_id, data_cache):

        if not data_cache or sample_id is None:
            print("COND 1")
            return "", {'display': 'none'}

        if not data_cache["outputs"]:
            print("COND 2")
            print("SCORES", data_cache["outputs"])
            return "", {'display': 'none'}

        tasks  = data_cache["exp_config"]["tasks"]
        print("HIGHLIFHGT!!!!!", "seg" in data_cache["exp_config"]["tasks"], data_cache["exp_config"]["tasks"])
        if "seg" not in tasks:
            return "", {'display': 'none'}

        task2labels = data_cache["exp_config"]["task2label"]
        outputs = data_cache["outputs"]

        print("SAMPLE KEYS", outputs.keys())

        sample_out = outputs[sample_id]

        print("SAMPLE OUT HIGHLKIGHT", sample_out)

        if "ac" in sample_out["preds"]:
            task = "ac"
        else:
            task = "seg"

        data = []
        for i,token in enumerate(sample_out["text"]):

            """
            EXAMPLE DATA for one token
            {
                    "token": "that",
                    "pred": {
                                "span_id": None,
                                "label": "X",
                                "score": 0.1,

                                },
                    "gold": {
                                "span_id": "X_1",
                                "label": "X",
                            }

                },
            """

            pred_span = sample_out["preds"]["span_ids"][i] if "span" in sample_out["preds"]["span_ids"][i] else None
            gold_span = sample_out["gold"]["span_ids"][i] if "span" in sample_out["gold"]["span_ids"][i] else None

            if pred_span:
                pred_label = sample_out["preds"][task][i]

            if task in sample_out["probs"]:
                scores = sample_out["probs"][task][i]
                idx = np.argmax(scores)
                score = scores[idx]
                #will be the same as pred_label
                pred_label = task2labels[task][idx]

            token_data = {
                            "token": token,
                            "pred":{
                                    "span_id": pred_span,
                                    "label": pred_label,
                                    "score": score,

                                        }
                            }
            
            if gold_span:
                pred_label = sample_out["gold"][tasks]
                token_data["gold"] ={
                                    "span_id":gold_span,
                                    "label":pred_label
                                    }


            data.append(token_data)

        print("OK WE GOT HERE FOR HIGHLIGHT")
        hotviz.hot_text(data, labels=task2labels[task], save_path=th_path)

        with open(th_path, "rb") as f:
            enc_img = base64.b64encode(f.read())
            src = f"data:image/png;base64,{enc_img.decode()}"

        img =  html.Img(   
                        id="text-highlight",
                        src=src
                        )

        return img, {'display': 'block'}


    def update_tree_graph(self, sample_id, data_cache, fig_state):


        def extract_data(df):
            spans = df.groupby("span_ids")
            data = []
            for i, span in spans:
                span_data.append({
                                    "label":span["ac"].unique()[0],
                                    "link": span["relation"].unique()[0],
                                    "link_label": span["stance"].unique()[0],
                                    "text": " ".join(span["text"].tolist()),
                                    })
            
            return data

        fig_state = go.Figure(fig_state)

        if not data_cache or sample_id is None:
            return fig_state, {'display': 'none'}

        if not data_cache["scores"]:
            return fig_state, {"display":"none"}

        if "relation" not in data_cache["exp_config"]["tasks"]:
            return go.Figure([]), {'display': 'none'}

        outputs = data_cache["outputs"]

        if not outputs:
            return []

        sample_out = outputs[sample_id]

        pred_df = pd.DataFrame(sample_out["preds"])
        gold_df = pd.DataFrame(sample_out["gold"])
        text = [" ".join(t) for t in sample_out["text"]]

        if "span_ids" in pred_df.columns:
            pred_data = extract_data(pred_df)
            gold_data = extract_data(gold_df)
      
        else:
            rename_dict = {"ac": "label", "relation": "link", "stance":"link_label"}
            pred_df.rename(columns=rename_dict,  inplace=True)
            gold_df.rename(columns=rename_dict,  inplace=True)
            pred_df["text"] = text
            gold_df["text"] = text
            pred_data = list(pred_df.loc[:,list(rename_dict.values())+["text"]].T.to_dict().values())
            gold_data = list(gold_df.loc[:,list(rename_dict.values())+["text"]].T.to_dict().values())


        # example input:
        # """
        #     [{   
        #     'label': 'MajorClaim',
        #     'link': 1,
        #     'link_label': '',
        #     'text': 'one who studies overseas will gain many skills throughout this '
        #             'experience'
        #             },]
        # """

        fig = hotviz.hot_tree(pred_data, gold_data=gold_data)

        last_vis_state = get_visible_info(fig_state)
        current_vis_state = get_visible_info(fig)
        if last_vis_state.keys() == current_vis_state.keys():
            update_visible_info(fig, last_vis_state)


        return fig, {'display': 'block'}


    def update_conf_dropdown(self, experiment_id):
        exp_config = self.db.get_exp_config({"experiment_id":experiment_id})

        if exp_config is None:
            return [], None

        options = [{"label":t, "value":t} for t in exp_config["tasks"]]
        value = options[0]["value"]


        return options, value


    def update_conf_matrix(self, task, data_cache):

        if not data_cache or task is None:
            return {}, {"display":"none"}
        
        if not data_cache["scores"]:
            return {}, {"display":"none"}

        df = pd.DataFrame(data_cache["scores"])
        
        cond1 = df["epoch"] == data_cache["epoch"]
        cond2 = df["split"] == "val"
        conf_m = np.array(df[cond1 & cond2][f"{task}-confusion_matrix"].to_numpy()[0])

        conf_m = np.round(conf_m / np.sum(conf_m), 2)
        labels = data_cache["exp_config"]["dataset_config"]["task_labels"][task]

        fig = conf_matrix(conf_m, labels)

        return fig, {'display': 'block'}


    def update_rank_graph(self, clickData, project, task, rank_split):


        if None in [project, task, rank_split]:
            return go.Figure([]), {"display":"none"}


        rank_metric = "f1"
        top_n = 10
        #dataset_name, project, rank_task, rank_metric, rank_split, top_n = rank_values

        filter_by = get_filter(project=project)
        experiments = pd.DataFrame(db.get_experiments(filter_by))
        experiment_ids = list(experiments["experiment_id"].to_numpy())
        data = db.get_scores( {"experiment_id":{"$in": experiment_ids}})

        experiment2config = {}
        for i, exp_row in experiments.iterrows():
            exp_id = exp_row["experiment_id"]
            config = exp_row.to_dict()
            config.pop("_id")
            experiment2config[exp_id] = config
        
        #print(list(experiment2config.items())[-1][1])
        #NOTE! we are assuming that all experiments are done with the same metrics
        # and we are displaying only
        display_splits = ["val", "test"] 
        display_metrics = list(experiment2config.items())[-1][1]["metrics"]

        
        score_data = []
        exp_groups = data.groupby("experiment_id")
        for exp_id in exp_groups.groups.keys():

            exp_split_group = exp_groups.get_group(exp_id).groupby("split")

            test_model_choice = experiment2config[exp_id]["model_selection"]
            exp_val_data = exp_split_group.get_group("val")
            if test_model_choice == "best":
                val_row = exp_val_data.sort_values(rank_metric, ascending=False).head(1)
            else:
                val_row = exp_val_data.sort_values("epoch", ascending=False).head(1)


            test_exist = False
            if "test" in exp_split_group.groups:
                test_row = exp_split_group.get_group("test").head(1)
                test_exist = True

            exp_score = {"experiment_id":exp_id}
            for metric in display_metrics:
                task_metric = f"{task}-{metric}"

                exp_score[f"val-{task_metric}"] = val_row[task_metric].to_numpy()[0]

                if test_exist:
                    exp_score[f"test-{task_metric}"] = test_row[task_metric].to_numpy()[0]
                else:
                    exp_score[f"test-{task_metric}"] = 0

                # if "val" in display_splits:
                #     exp_score[f"val-{task_metric}"] = val_row[task_metric].to_numpy()[0]

            score_data.append(exp_score)


        ascend = True if "loss" == rank_metric else False
        score_df = pd.DataFrame(score_data)
        score_df.sort_values(f"{rank_split}-{task}-{rank_metric}", inplace=True, ascending=ascend)

        top_scores = score_df.head(top_n)
        top_scores.reset_index(drop=True, inplace=True)

        fig = rank_bar(top_scores, [rank_metric], [rank_split], experiment2config, rank_task, top_n, clickData)
        
        return fig, {'display': 'block'}