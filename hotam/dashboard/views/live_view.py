
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

class LiveView:

    def __init__(self, app, db):
        self.db = db
        

        self.id2exp = dict(enumerate(db.get_live_exps()))

        buttons = html.Div([
                            html.Button(
                                        id='exp-btn-{}',
                                        children=exp["experiment_id"],
                                        n_clicks=0
                                        ) 
                            for i,exp in self.id2exp.items()
                            ]
                            +
                            [ html.Div(
                                        id='experiment-id',
                                        children=''
                                        )
                            ]
                            )
                            #html.Div(id='container-button-timestamp')

        self.layout =   html.Div(   
                                className="row flex-display",
                                children=[
                                            html.Div(
                                                    children=[
                                                            daq.ToggleSwitch(
                                                                            id='my-toggle-switch',
                                                                            label="Hide Stats",
                                                                            #vertical=True,
                                                                            value=False
                                                                            ),
                                                             ]),
                                            self.__stats(),
                                            html.Div(
                                                        id="visuals"
                                                        ),
                                            html.Div(id='data-cache', children=dict(), style={'display': 'none'}),
                                        ],
                                style={"display": "flex", "flex-direction": "column"},
                            )

        ### CALLBACKS
        app.callback(
                    Output('data-cache', 'children'),
                    Input('interval-component', 'n_intervals')],
                    [State('data-cache', 'children')])(self.update_data_cache)


        app.callback(Output('experiment-id', 'children'),
                    [Input(f'exp-btn-{}', 'n_clicks')
                    for i in self.live_exps.keys() ])(self.get_exp)

                        
        app.callback(
                Output('exp-config', 'children'),
                [Input("config-dropdown2", "value"),
                Input('data-cache', 'children')])(self.get_config)


        #data table
        app.callback(
                    Output('data-table', 'figure'),
                    [Input('data-cache', 'children')]
                    )(self.get_data_table)

        #visuals
        app.callback(Output('visuals', 'children'),
                    [
                        Input('data-cache', 'children'),
                        #Input("task-checklist", 'value'),
                        #Input("metric-checklist", 'value')
                        ])(self.update_visuals)


    def __stats(self, settings=None):
        return html.Div(
                        className="pretty_container fourteen columns",
                        id="stats",
                        children=[
                                    #html.Button('Hide', id='hide-settings-info', n_clicks=0),
                                    html.Div(
                                            className="row flex-display",
                                            children=[
                                                        html.Div(  
                                                            className="pretty_container",
                                                            children=[
                                                                        dcc.Dropdown(
                                                                                    id='config-dropdown2',
                                                                                    options=[
                                                                                                {"label": "hyperparamaters", "value":"hyperparamaters"},
                                                                                                {"label": "exp config", "value":"dataset_config"},
                                                                                                {"label": "trainer_args", "value":"trainer_args"}
                                                                                            ],
                                                                                    value="hyperparamaters",
                                                                                    className="dcc_control",
                                                                                    ),
                                                                        html.Div( 
                                                                                className="pretty_container",
                                                                                children=[
                                                                                            html.Pre(
                                                                                                    id="exp-config2",
                                                                                                    children=""
                                                                                                    )
                                                                                            ],
                                                                                style={
                                                                                        "maxHeight": "500px", 
                                                                                        "maxWidth": "300px",
                                                                                        "overflow": "scroll",
                                                                                        }
                                                                                )
                                                                        ]       
                                                            ),
            
                                                        html.Div( 
                                                                className="pretty_container five columns",
                                                                children=[
                                                                            dcc.Graph(
                                                                                        id='data-table',
                                                                                        figure = fig_layout(go.Figure(data=[])),
                                                                                )
                                                                            
                                                                        ]
                                                                    ),

                                                    ]
                                            ),
                                ]
                        )



    def get_exp(self,*args):

        for i, click in enumerate(args):
        
            if click == 1:
                return self.




    def update_output(self, value):
        if value:
            return {'display': 'none'}
        else:
            return {'display': 'block'}


    def update_data_cache(self, experiment_id, n, cache_state):

        
        prev_epoch = cache_state.get("epoch", -1)
        current_exp = cache_state.get("experiment_id")

        filter_by = get_filter(experiment=experiment_id)
        last_epoch = self.db.get_last_epoch(filter_by)

        if experiment_id == current_exp:
            if last_epoch == prev_epoch:
                return dash.no_update

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

        plot_info = {
                    "class-metric-graph":{
                                            "visibility":{}
                                            }, 
                    "task-metric-graph":{
                                            "visibility":{}
                                            }, 
                    "loss-graph":{
                                            "visibility":{}
                                            }, 
                    }
        if last_epoch != 0:
            plot_info["class-metric-graph"]["visibility"] = get_visible_info(State("class-metric-graph", "figure"))
            plot_info["task-metric-graph"]["visibility"] = get_visible_info(State("task-metric-graph", "figure"))
            plot_info["loss-graph"]["visibility"] = get_visible_info(State("loss-graph", "figure"))

        return {
                "exp_config": exp_config,
                "experiment_id": experiment_id,
                "epoch": last_epoch,
                "scores": scores,
                "outputs": outputs,
                "plot_info":plot_info,
                }


    def get_data_table(self, data_cache):
        data = data_cache["exp_config"]["dataset_stats"]
        df = pd.DataFrame(data)
        return make_table(df, "Dataset Statistics")


    def get_config(self, config_value, data_cache):
        exp_config = data_cache.get("exp_config", {})

        if config_value == "exp config":
            # exp_config.pop("trainer_args")
            # exp_config.pop("hyperparamaters")
            # exp_config.pop("_id")
            return json.dumps(exp_config.get("dataset_config",{}), indent=4)
        else:
            return json.dumps(exp_config.get(config_value,{}), indent=4)


    def loss_graph(self, data_cache):
        
        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]
        tasks = data_cache["exp_config"]["tasks"]
        
        task_loss = [task+"-loss" for task in tasks]

        figure = make_lineplot(data, task_loss, "Loss")

        visible_info = data_cache["plot_info"]["loss-graph"]["visibility"]
        if visible_info:
            update_visible_info(figure, visible_info)

        graph =  dcc.Graph(
                        id="loss-graph",
                        figure = fig_layout(figure)
                        )
        return graph


    def task_metric_graph(self, data_cache):

        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]
        
        tasks = data_cache["exp_config"]["tasks"]
        metrics = data_cache["exp_config"]["metrics"]

        task_metrics = []
        for task in tasks:
            for metric in metrics:
                task_metrics.append("-".join([task,metric]))
    
        figure = make_lineplot(data, task_metrics, "Task Scores")

        visible_info = data_cache["plot_info"]["task-metric-graph"]["visibility"]
        if visible_info:
            update_visible_info(figure, visible_info)


        graph =  dcc.Graph(
                        id="task-metric-graph",
                        figure = fig_layout(figure)
                        )
        
        return graph


    def class_metric_graph(self, data_cache):

        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]

        tasks = data_cache["exp_config"]["tasks"]
        metrics = data_cache["exp_config"]["metrics"]        
        task2labels = data_cache["exp_config"]["dataset_config"]["task_labels"]

        filter_columns = []
        for task in tasks:
            classes = task2labels[task]
            for c in classes:
                for metric in metrics:
                    filter_columns.append("-".join([task, c, metric]).lower())

        

        figure =  make_lineplot(data, filter_columns, "Class Scores")

        visible_info = data_cache["plot_info"]["class-metric-graph"]["visibility"]
        if visible_info:
            update_visible_info(figure, visible_info)

        graph =  dcc.Graph(
                        id="class-metric-graph",
                        figure = fig_layout(figure)
                        )
        return graph


    def highlight_text(self, data_cache, sample_id):

        outputs = data_cache["outputs"]

        if not outputs:
            return []

        task2labels = data_cache["exp_config"]["task2label"]
        sample_out = outputs[sample_id]

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


        hotviz.hot_text(data, labels=task2labels[task], save_path=th_path)

        with open(th_path, "rb") as f:
            enc_img = base64.b64encode(f.read())
            src = f"data:image/png;base64,{enc_img.decode()}"

        img =  html.Img(   
                        id="text-highlight",
                        src=src
                        )

        return img


    def tree_graph(self, data_cache, sample_id):

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

        outputs = data_cache["outputs"]

        if not outputs:
            return []

        sample_out = outputs[sample_id]

        pred_df = pd.DataFrame(sample_out["pred"])
        gold_df = pd.DataFrame(sample_out["gold"])

        if "span_ids" in pred_df.columns:
            pred_data = extract_data(pred_df)
            gold_data = extract_data(pred_df)
      
        else:
            rename_dict = {"ac": "label", "relation": "link", "stance":"link_label"}
            pred_df.rename(columns={"ac": "label", "relation": "link", "stance":"link_label"})
            gold_df.rename(columns={"ac": "label", "relation": "link", "stance":"link_label"})
            pred_data = pred_df.loc[:,[list(rename_dict.values())+["text"]]].to_dict()
            gold_data = gold_df.loc[:,[list(rename_dict.values())]+["text"]].to_dict()


        # example input:
        """
            [{   
            'label': 'MajorClaim',
            'link': 1,
            'link_label': '',
            'text': 'one who studies overseas will gain many skills throughout this '
                    'experience'
                    },]
        """

        fig = hotviz.hot_tree(pred_data, gold_data=gold_data)
        graph =  dcc.Graph(
                        id="tree-graph",
                        figure = fig_layout(figure)
                        )

        return graph


    def get_visual_box(self, key, data_cache):
        
        if key == "loss":
            graph =  [self.loss_graph(data_cache)]
        elif key == "task_metric":
            graph = [self.task_metric_graph(data_cache)]
        elif key == "class_metric":
            graph = [self.class_metric_graph(data_cache)]
        elif key == "tree":
            graph = [self.tree_graph(data_cache, 200)]
        elif key == "highlight":
            graph = [self.highlight_text(data_cache, 200)]

        box = html.Div(
                    className="pretty_container six columns",
                    children=graph)
        return box


    def update_visuals(self, data_cache):
        
        if not data_cache:
            return []

        visuals = ["task_metric", "class_metric", "loss"] #, "confusion_matrix"]

        tasks  = data_cache["exp_config"]["tasks"]
        if "seg" in tasks:
            visuals.append("highlight")
        
        if "ac" in tasks and "relation" in tasks and "stance" in tasks:
            visuals.append("tree")

        nr_rows = math.ceil(len(visuals)/2) * 2
        rows = []
        for i in range(0, nr_rows, 2):
            rows.append(html.Div(
                                className="row flex-display",
                                children=[
                                            self.get_visual_box(v, data_cache) 
                                            for v in visuals[i:i+2]
                                            ]        
                            ))
        return rows