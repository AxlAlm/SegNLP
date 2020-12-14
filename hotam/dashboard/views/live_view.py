
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

        # button_menu = html.Div(
        #                         id='exp-dropdown',
        #                         #children = []
        #                         )

        info = html.Div(
                        className="row flex-display",
                        children=[
                                    html.Div(  
                                        className="pretty_container six columns",
                                        children=[
                                                    dcc.Dropdown(
                                                                id='exp-config-dropdown',
                                                                options=[
                                                                            {"label": "hyperparamaters", "value":"hyperparamaters"},
                                                                            {"label": "exp config", "value":"dataset_config"},
                                                                            {"label": "trainer_args", "value":"trainer_args"}
                                                                        ],
                                                                value="hyperparamaters",
                                                                className="dcc_control",
                                                                ),
                                                    html.Div( 
                                                            className="pretty_container six columns",
                                                            children=[
                                                                        html.Pre(
                                                                                id="exp-config",
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
                                            className="pretty_container six columns",
                                            children=[
                                                        dcc.Graph(
                                                                    id='data-table',
                                                                    figure = go.Figure(data=[]),
                                                            )
                                                        
                                                    ]
                                                )

                                ]
                            )
    

        
        visuals =  html.Div(
                            children=[
                                html.Div(
                                    className="row flex-display",
                                    children=[
                                                html.Div(
                                                        id="loss-graph-con",
                                                        className="pretty_container six columns",
                                                        children=[dcc.Graph(
                                                                            id="loss-graph",
                                                                            figure = go.Figure([])
                                                                            )
                                                                    ],
                                                        style={'display': 'none'}
                                                        ),
                                                html.Div(
                                                        id="task-metric-graph-con",
                                                        className="pretty_container six columns",
                                                        children=[dcc.Graph(
                                                                            id="task-metric-graph",
                                                                            figure = go.Figure([])
                                                                            )
                                                                    ],                                                    
                                                        style={'display': 'none'}
                                                        ),
                                                ]
                                        ),
                                html.Div(
                                    className="row flex-display",
                                    children=[
                                                html.Div(
                                                        id="class-metric-graph-con",
                                                        className="pretty_container six columns",
                                                        children=[dcc.Graph(
                                                                            id="class-metric-graph",
                                                                            figure = go.Figure([])
                                                                            )
                                                                            ],                                                      
                                                        style={'display': 'none'}
                                                        ),
                                                html.Div(
                                                        id="conf-matrix-con",
                                                        className="pretty_container six columns",
                                                        children=[
                                                                    dcc.Dropdown(
                                                                                id='conf-dropdown',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                ),
                                                                    dcc.Graph(
                                                                                id="conf-matrix",
                                                                                figure = go.Figure([])
                                                                                )
                                                                    ],                                                       
                                                        style={'display': 'none'}
                                                        ),
                                                ]
                                        ),
                                html.Div(
                                    className="row flex-display",
                                    children=[

                                                html.Div(
                                                        id="text-highlight-con",
                                                        className="pretty_container six columns",
                                                        children=[ 
                                                                    dcc.Dropdown(
                                                                                id='sample-id-dropdown1',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                ),
                                                                    html.Img(   
                                                                            id="text-highlight",
                                                                            src=""
                                                                            ),
                                                                    ],
                                                        style={'display': 'none'}
                                                        ),
                                                html.Div(
                                                        id="tree-graph-con",
                                                        className="pretty_container six columns",
                                                        children=[
                                                                    dcc.Dropdown(
                                                                                id='sample-id-dropdown2',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                ),
                                                                    dcc.Graph(
                                                                            id="tree-graph",
                                                                            figure = go.Figure([])
                                                                            )
                                                                ],                                                         
                                                        style={'display': 'none'}
                                                        ),
                                                ]
                                        )
                                ]
                            )
            
        live_view = html.Div(   
                            id="live-view",
                            #className="row flex-display",
                            children=[
                                        info,
                                        visuals
                                        ],
                            style={'display': 'none'}
                            #style={"display": "flex", "flex-direction": "column"},
                            )


        self.layout = html.Div(
                                children=[
                                            dcc.Dropdown(
                                                        id='exp-dropdown',
                                                        options=[],
                                                        value=None,
                                                        className="dcc_control",
                                                        #clearable=False,
                                                        ),
                                            live_view,
                                            html.Div(id='data-cache', children=dict(), style={'display': 'none'}),

                                          ]
                                
                                )

        
        app.callback(Output('exp-dropdown', 'options'),
                    Input('interval-component', 'n_intervals'),
                    State('exp-dropdown', 'value'))(self.update_exp_dropdown)


        app.callback(
                    Output('data-cache', 'children'),
                    Output('live-view', 'style'),
                    [Input('interval-component', 'n_intervals'),
                    Input('exp-dropdown','value')
                    ],
                    [State('data-cache', 'children')])(self.update_data_cache)

            
        app.callback(Output('exp-config', 'children'),
                    [Input("exp-config-dropdown", "value"),
                    Input('data-cache', 'children')])(self.get_config)

        app.callback(Output('data-table', 'figure'),
                    [Input('data-cache', 'children')])(self.get_data_table)



        app.callback(
                    Output('loss-graph', 'figure'),
                    Output('loss-graph-con', 'style'),
                    [Input('data-cache', 'children')],
                    [State('loss-graph', 'figure')])(self.update_loss_graph)


        app.callback(
                    Output('task-metric-graph', 'figure'),
                    Output('task-metric-graph-con', 'style'),
                    [Input('data-cache', 'children')],
                    [State('task-metric-graph', 'figure')])(self.update_task_metric_graph)


        app.callback(
                    Output('class-metric-graph', 'figure'),
                    Output('class-metric-graph-con', 'style'),
                    [Input('data-cache', 'children')],
                    [State('class-metric-graph', 'figure')])(self.update_class_metric_graph)


        app.callback(Output('conf-dropdown', 'options'),
                    Output('conf-dropdown', 'value'),
                    Input('exp-dropdown','value')
                    )(self.update_conf_dropdown)


        app.callback(
                    Output('conf-matrix', 'figure'),
                    Output('conf-matrix-con', 'style'),
                    [
                    Input('conf-dropdown', 'value'),
                    Input('data-cache', 'children')
                    ])(self.update_conf_matrix)

        
        app.callback(Output('sample-id-dropdown1', 'options'),
                    Output('sample-id-dropdown1', 'value'),
                    Input('exp-dropdown','value')
                    )(self.update_sample_id_dropdown)


        app.callback(
                    Output('text-highlight', 'src'),
                    Output('text-highlight-con', 'style'),
                    [
                    Input('sample-id-dropdown1', 'value'),
                    Input('data-cache', 'children')
                    ])(self.update_highlight_text)


        app.callback(Output('sample-id-dropdown2', 'options'),
                    Output('sample-id-dropdown2', 'value'),
                    Input('exp-dropdown','value')
                    )(self.update_sample_id_dropdown)

        app.callback(
                    Output('tree-graph', 'figure'),
                    Output('tree-graph-con', 'style'),
                    [
                    Input('sample-id-dropdown2', 'value'),
                    Input('data-cache', 'children')
                    ],
                    [State('tree-graph', 'figure')])(self.update_tree_graph)


    def update_exp_dropdown(self, n):
        exps = sorted(self.db.get_live_exps_ids())
        return [{"label":e, "value":e} for e in exps]

        
    def update_output(self, value):
        if value:
            return {'display': 'none'}
        else:
            return {'display': 'block'}


    def update_data_cache(self, n, experiment_id,  cache_state):
   
        if experiment_id is None and not cache_state:
            return dash.no_update

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

        data_cache = {
                "exp_config": exp_config,
                "experiment_id": experiment_id,
                "epoch": last_epoch,
                "scores": scores,
                "outputs": outputs,
                }
        
        return data_cache, {"display":"block"}


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
                    filter_columns.append("-".join([task, c, metric]).lower())

    
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

        options = [{"label":t, "value":t} for t in exp_config["dataset_config"]["tracked_sample_ids"]]
        value = options[0]["value"]
        print("SAMPLE ID")
        return options, value


    def update_highlight_text(self, sample_id, data_cache):

        if not data_cache or sample_id is None:
            return "", {'display': 'none'}

        if not data_cache["outputs"]:
            return "", {'display': 'none'}

        tasks  = data_cache["exp_config"]["tasks"]
        if "seg" not in tasks:
            return "", {'display': 'none'}

        task2labels = data_cache["exp_config"]["task2label"]
        outputs = data_cache["outputs"]

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

        return img, {'display': 'block'}


    def update_tree_graph(self, sample_id, data_cache, fig_state):

        fig_state = go.Figure(fig_state)

        if not data_cache or sample_id is None:
            return fig_state, {'display': 'none'}

        if not data_cache["scores"]:
            return fig_state, {"display":"none"}


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


        if "relation" not in data_cache["exp_config"]["tasks"]:
            return go.Figure([]), {'display': 'none'}

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


        last_vis_state = get_visible_info(fig_state)
        current_vis_state = get_visible_info(figure)
        if last_vis_state.keys() == current_vis_state.keys():
            update_visible_info(figure, last_vis_state)


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

        last_epoch_data = [d for d  in data_cache["scores"] if d["epoch"] == data_cache["epoch"]][0]
        conf_data = last_epoch_data["confusion_matrix"][task]

        print(conf_data)

        fig = make_table(df, "Confusion Matrix")
        return fig, {'display': 'block'}