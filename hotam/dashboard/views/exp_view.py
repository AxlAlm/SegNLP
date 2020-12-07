
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
from hotam.dashboard.utils import get_filter, fig_layout
from hotam.dashboard.views.visuals import *


th_path  = "/tmp/text_highlight.png"

class ExpView:

    def __init__(self, app, db):
        self.db = db
        settings = self.__get_settings()
        info_row = self.__get__info_row(settings=settings)
        self.layout = self.__setup_view(info_row)


        ### CALLBACKS
        app.callback(
                    Output('data-cache', 'children'),
                    [Input('experiment-dropdown', 'value'),
                    Input('interval-component', 'n_intervals')],
                    [State('data-cache', 'children')])(self.update_data_cache)

        app.callback(
                    Output("setting-info", 'style'),
                    [Input('my-toggle-switch', 'value')])(self.update_output)

        app.callback(
                Output('exp-config2', 'children'),
                [Input("config-dropdown2", "value"),
                Input('data-cache', 'children')])(self.get_config)


        ## Dropdowns
        app.callback(
                Output('project-dropdown', 'options'),
                [Input('dataset-dropdown', 'value')])(self.update_project_dropdown)

        app.callback(
                Output('dataset-dropdown', 'options'),
                [Input('project-dropdown', 'value')])(self.update_dataset_dropdown)
        
        app.callback(
                Output('model-dropdown', 'options'),
                [Input('dataset-dropdown', 'value'),
                Input('project-dropdown', 'value')])(self.update_model_dropdown)

        #checklist
        app.callback(
                Output("metric-checklist", 'options'),
                [Input('experiment-dropdown', 'value')])(self.get_metric_checklist)

        app.callback(
                Output("task-checklist", 'options'),
                [Input('experiment-dropdown', 'value')])(self.get_task_checklist)

        app.callback(
                Output("class-checklist", 'options'),
                [Input("task-checklist", 'value'),  
                Input('experiment-dropdown', 'value')])(self.get_class_checklist)


        #plots
        app.callback(
                    Output('data-table', 'figure'),
                    [Input('data-cache', 'children')]
                    )(self.get_data_table)


        app.callback(Output('box1', 'children'),
                    [
                        Input('box1-dropdown', 'value'),
                        Input('data-cache', 'children'),

                        Input("task-checklist", 'value'),
                        Input("metric-checklist", 'value')
                        ])(self.get_visual_box)

        app.callback(Output('box2', 'children'),
                    [
                        Input('box2-dropdown', 'value'),
                        Input('data-cache', 'children'),

                        Input("task-checklist", 'value'),
                        Input("metric-checklist", 'value'),
                        ])(self.get_visual_box)


        app.callback(Output('box3', 'children'),
                    [
                        Input('box3-dropdown', 'value'),
                        Input('data-cache', 'children'),

                        Input("task-checklist", 'value'),
                        Input("metric-checklist", 'value'),

                        ])(self.get_visual_box)

        app.callback(Output('box4', 'children'),
                    [
                        Input('box4-dropdown', 'value'),
                        Input('data-cache', 'children'),

                        Input("task-checklist", 'value'),
                        Input("metric-checklist", 'value'),
                        ])(self.get_visual_box)


    def __get_settings(self):
        return html.Div(  
                                #className='two columns div-user-controls',
                                className="pretty_container three columns",
                                #className="row flex-display",
                                children=[
                                            html.H3('Settings'),
                                            html.P("Dataset:", className="control_label"),
                                            dcc.Dropdown(
                                                        id='dataset-dropdown',
                                                        options=self.db.get_dropdown_options("dataset"),
                                                        value=self.db.get_last_exp().get("dataset",[]),
                                                        className="dcc_control",
                                                    ),
                                            html.P("Project:", className="control_label"),
                                            dcc.Dropdown(
                                                        id='project-dropdown',
                                                        options=self.db.get_dropdown_options("project"),
                                                        value=self.db.get_last_exp().get("project",[]),
                                                        className="dcc_control",
                                                    ), 
                                            html.P("Model:", className="control_label"),
                                            dcc.Dropdown(
                                                        id='model-dropdown',
                                                        options=self.db.get_dropdown_options("model"),
                                                        value=self.db.get_last_exp().get("model",[]),
                                                        className="dcc_control",
                                                    ),
                                            html.P("Experiment ID:", className="control_label"),
                                            dcc.Dropdown(
                                                        id='experiment-dropdown',
                                                        options=self.db.get_dropdown_options("experiment_id"),
                                                        style={'height': '30px', 'width': '300px'},
                                                        value=self.db.get_last_exp().get("experiment_id",[]),
                                                        className="dcc_control",
                                                    ),
                                            html.P("Filter by Task:", className="control_label"),
                                            dcc.Checklist(
                                                id="task-checklist",
                                                options=[],
                                                value=self.db.get_last_exp().get("tasks",{}),
                                                labelStyle={'display': 'inline-block'},
                                                className="dcc_control",
                                            ),
                                            html.P("Filter by Metric:", className="control_label"),
                                            dcc.Checklist(
                                                id="metric-checklist",
                                                options=[],
                                                value=["f1"], #db.get_last_exp().get("metrics",[]),
                                                labelStyle={'display': 'inline-block'},
                                                className="dcc_control",
                                            ),
                                            html.P("Filter by Class:", className="control_label"),
                                            dcc.Checklist(
                                                id="class-checklist",
                                                options=[],
                                                value=[l for task_labels in list(self.db.get_last_exp().get("task2label", {}).values()) for l in task_labels],
                                                labelStyle={'display': 'inline-block'},
                                                className="dcc_control",
                                            ),

                                            ]
                        )


    def __get__info_row(self, settings=None):
        return html.Div(
                        className="pretty_container fourteen columns",
                        id="setting-info",
                        children=[
                                    #html.Button('Hide', id='hide-settings-info', n_clicks=0),
                                    html.Div(
                                            className="row flex-display",
                                            children=[
                                                        settings,
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



    def __setup_view(self, info_row):
        visuals = ["loss", "tree", "highlight", "task_metric", "class_metric"]
        return  html.Div(   
                                #className='row',  # Define the row element
                                className="row flex-display",
                                children=[
                                            html.Div(
                                                    children=[
                                                            daq.ToggleSwitch(
                                                                id='my-toggle-switch',
                                                                label="Hide Settings and Info",
                                                                #vertical=True,
                                                                value=False
                                                                ),
                                                    ]),
                                            info_row,
                                            html.Div(
                                                        id="visuals"
                                                        ),
                                            # self.__create_box_row(1),
                                            # self.__create_box_row(3),
                                            html.Div(id='data-cache', children=dict(), style={'display': 'none'}),
                                        ],
                                style={"display": "flex", "flex-direction": "column"},
                            )
        


    # def __create_box_row(self, id_start):
    #     i = id_start
    #     i2 = id_start+1
    #     visuals = ["loss", "tree", "highlight", "task_metric", "class_metric"]
    #     row = html.Div(
    #                     className="row flex-display",
    #                     children=[
    #                                 html.Div(
    #                                     className="pretty_container six columns",
    #                                     children=[
    #                                                 dcc.Dropdown(
    #                                                             id=f'box{i}-dropdown',
    #                                                             options=[{"label":v, "value":v} for v in visuals],
    #                                                             value=[],
    #                                                             className="dcc_control",
    #                                                             ),
    #                                                 html.Div(id=f"box{i}")
    #                                             ]
    #                                             ),
    #                                 html.Div(
    #                                     className="pretty_container six columns",
    #                                     children=[ 
    #                                                 dcc.Dropdown(
    #                                                             id=f'box{i2}-dropdown',
    #                                                             options=[{"label":v, "value":v} for v in visuals],
    #                                                             value=[],
    #                                                             className="dcc_control",
    #                                                             ),
    #                                                 html.Div(id=f"box{i2}")
    #                                             ],
    #                                 ),
    #                             ],
    #                 )
    #     return row

    def update_output(self, value):
        if value:
            return {'display': 'none'}
        else:
            return {'display': 'block'}


    def update_project_dropdown(self, v):
        return self.update_dropdown("project", dataset=v)


    def update_dataset_dropdown(self, v):
        return self.update_dropdown("dataset", project=v)


    def update_model_dropdown(self, v, v1):
        return self.update_dropdown("model", dataset=v, project=v1)


    def update_experiment_dropdown(self, v, v1, v2):
        return self.update_dropdown("experiment_id", dataset=v, project=v1, model=v2)


    def update_dropdown(self, search_for, dataset=None, project=None, model=None):
        filter_by = get_filter(dataset=dataset, project=project, model=model)
        return self.db.get_dropdown_options(search_for, filter_by=filter_by)


    def get_metric_checklist(self, experiment_id):
        return self.db.get_metrics(get_filter(experiment=experiment_id))


    def get_task_checklist(self, experiment_id):
        return self.db.get_tasks(get_filter(experiment=experiment_id))


    def get_class_checklist(self, tasks, experiment_id):
        return self.db.get_task_classes(get_filter(experiment=experiment_id), tasks=tasks)


    def update_data_cache(self, experiment_id, n, cache_state):

        current_epoch = cache_state.get("epoch", -1)
        current_exp = cache_state.get("experiment_id")

        filter_by = get_filter(experiment=experiment_id)
        last_epoch = self.db.get_latest_epoch(filter_by)

        if experiment_id == current_exp:
            if last_epoch == current_epoch:
                return dash.no_update

        exp_config = self.db.get_exp(experiment_id)
        exp_config.pop("_id")

        filter_by["epoch"] =  { "$lte": last_epoch}
        scores = self.db.get_scores(filter_by)
        current_epoch = scores.epoch.max()
        scores.pop("_id")

        outputs = self.db.get_outputs(filter_by)
        outputs.pop("_id")

        return {
                "exp_config": exp_config,
                "experiment_id": experiment_id,
                "epoch": current_epoch,
                "scores": scores.to_dict(),
                "outputs": outputs
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


    def loss_graph(self, data_cache, tasks):
        
        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]
        
        task_loss = [task+"-loss" for task in tasks]

        figure = make_lineplot(data, task_loss, "Loss")
        graph =  dcc.Graph(
                        id="loss-graph",
                        figure = fig_layout(figure)
                        )
        return graph


    def task_metric_graph(self, data_cache, tasks, metrics):

        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]

        task_metrics = []
        for task in tasks:
            for metric in metrics:
                task_metrics.append("-".join([task,metric]))
    
        figure = make_lineplot(data, task_metrics, "Task Scores")
        graph =  dcc.Graph(
                        id="metric-graph",
                        figure = fig_layout(figure)
                        )
        
        return graph


    def class_metric_graph(self, data_cache, tasks, metrics):

        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]
        task2labels = data_cache["exp_config"]["dataset_config"]["task_labels"]

        filter_columns = []
        for task in tasks:
            classes = task2labels[task]
            for c in classes:
                for metric in metrics:
                    filter_columns.append("-".join([task, c, metric]).lower())

        figure =  make_lineplot(data, filter_columns, "Class Scores")
        graph =  dcc.Graph(
                        id="metric-graph",
                        figure = fig_layout(figure)
                        )
        return graph


    def highlight_text(self, data_cache, sample_id):

        outputs = data_cache["outputs"]
        task2labels = data_cache["exp_config"]["task2label"]
        sample_out = outputs[sample_id]

        # if "seg-ac" in sample_out["preds"]:
        #     task = "seg-ac" 
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


        src = hotviz.hot_text(data, labels=task2labels[task])
        
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


    def get_visual_box(self, key, cache, tasks, metrics):
        
        if key == "loss":
            graph =  [self.loss_graph(cache, tasks)]
        elif key == "task_metric":
            graph = [self.task_metric_graph(cache, tasks, metrics)]
        elif key == "class_metric":
            graph = [self.class_metric_graph(cache, tasks, metrics)]
        elif key == "tree":
            graph = [self.tree_graph(cache)]
        elif key == "highlight":
            graph = [self.highlight_text(cache)]

        box = html.Div(
                    className="pretty_container six columns",
                    children=graph)
        return box


def update_visuals(self, data_cache, tasks, metrics):

    visuals = ["task_metric", "class_metric", "loss"] #, "confusion_matrix"]

    tasks  = data_cache["exp_config"]["tasks"]
    if "seg" in tasks:
       visuals.append("highlight")
    
    if "ac" in tasks and "relation" in tasks and "stance" in tasks:
        visuals.append("tree")

    nr_rows = math.ceil(len(visuals)/2)
    rows = []
    for i in range(rows):
        rows.append(html.Div(
                            className="row flex-display",
                            children=[self.get_visual_box(visuals.pop(0), cache, tasks, metrics) for visual in range(2)]        
                        ))
    return rows