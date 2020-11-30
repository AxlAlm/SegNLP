
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

#import app and mongodb
from src.servers import app, db

#utils
from src.utils import get_filter, dropdown_filter, fig_layout

from src.visuals import *

th_path  = "/tmp/text_highlight.png"

settings  = html.Div(  
                        #className='two columns div-user-controls',
                        className="pretty_container three columns",
                        #className="row flex-display",
                        children=[
                                    html.H3('Settings'),
                                    html.P("Dataset:", className="control_label"),
                                    dcc.Dropdown(
                                                id='dataset-dropdown',
                                                options=db.get_dropdown_options("dataset"),
                                                value=db.get_last_exp().get("dataset",[]),
                                                className="dcc_control",
                                            ),
                                    html.P("Project:", className="control_label"),
                                    dcc.Dropdown(
                                                id='project-dropdown',
                                                options=db.get_dropdown_options("project"),
                                                value=db.get_last_exp().get("project",[]),
                                                className="dcc_control",
                                            ), 
                                    html.P("Model:", className="control_label"),
                                    dcc.Dropdown(
                                                id='model-dropdown',
                                                options=db.get_dropdown_options("model"),
                                                value=db.get_last_exp().get("model",[]),
                                                className="dcc_control",
                                            ),
                                    html.P("Experiment ID:", className="control_label"),
                                    dcc.Dropdown(
                                                id='experiment-dropdown',
                                                options=db.get_dropdown_options("experiment_id"),
                                                style={'height': '30px', 'width': '300px'},
                                                value=db.get_last_exp().get("experiment_id",[]),
                                                className="dcc_control",
                                            ),
                                    html.P("Filter by Task:", className="control_label"),
                                    dcc.Checklist(
                                        id="task-checklist",
                                        options=[],
                                        value=db.get_last_exp().get("tasks",{}),
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
                                        value=[l for task_labels in list(db.get_last_exp().get("task2label", {}).values()) for l in task_labels],
                                        labelStyle={'display': 'inline-block'},
                                        className="dcc_control",
                                    ),

                                    ]
                        )


info_row = html.Div(
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



def create_box_row(id_start):
    i = id_start
    i2 = id_start+1
    visuals = ["loss", "tree", "highlight", "task_metric", "class_metric"]
    row = html.Div(
                    className="row flex-display",
                    children=[
                                html.Div(
                                    className="pretty_container six columns",
                                    children=[
                                                dcc.Dropdown(
                                                            id=f'box{i}-dropdown',
                                                            options=[{"label":v, "value":v} for v in visuals],
                                                            value=[],
                                                            className="dcc_control",
                                                            ),
                                                html.Div(id=f"box{i}")
                                            ]
                                            ),
                                html.Div(
                                    className="pretty_container six columns",
                                    children=[ 
                                                dcc.Dropdown(
                                                            id=f'box{i2}-dropdown',
                                                            options=[{"label":v, "value":v} for v in visuals],
                                                            value=[],
                                                            className="dcc_control",
                                                            ),
                                                html.Div(id=f"box{i2}")
                                            ],
                                ),
                            ],
                )
    return row


exp_sides = html.Div(   
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
                                    create_box_row(1),
                                    create_box_row(3),
                                    html.Div(id='data-cache', children=dict(), style={'display': 'none'}),
                                ],
                        style={"display": "flex", "flex-direction": "column"},
                    )


@app.callback(
    Output("setting-info", 'style'),
    [Input('my-toggle-switch', 'value')])
def update_output(value):
    if value:
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@app.callback(
    Output('project-dropdown', 'options'),
    [Input('dataset-dropdown', 'value')])
def update_task_dropdown(v):
    return dropdown_filter("project", dataset=v)


@app.callback(
    Output('dataset-dropdown', 'options'),
    [Input('project-dropdown', 'value')])
def update_dataset_dropdown(v):
    return dropdown_filter("dataset", project=v)


@app.callback(
    Output('model-dropdown', 'options'),
    [Input('dataset-dropdown', 'value'),
    Input('project-dropdown', 'value')])
def update_model_dropdown(v,v1):
    return dropdown_filter("model", dataset=v, project=v1)


@app.callback(
    Output('experiment-dropdown', 'options'),
    [Input('dataset-dropdown', 'value'),
    Input('project-dropdown', 'value'),
    Input('model-dropdown', 'value')])
def update_experiment_dropdown(v,v1, v2):
    return dropdown_filter("experiment_id", dataset=v, project=v1, model=v2)


@app.callback(
    Output("metric-checklist", 'options'),
    [          
    Input('experiment-dropdown', 'value'),
    ])
def get_metric_checklist(experiment_id):
    metrics = db.get_metrics(get_filter(experiment=experiment_id))
    return metrics


@app.callback(
    Output("task-checklist", 'options'),
    [          
    Input('experiment-dropdown', 'value'),
    ])
def get_task_checklist(experiment_id):
    tasks = db.get_tasks(get_filter(experiment=experiment_id))
    return tasks


@app.callback(
    Output("class-checklist", 'options'),
    [        
    Input("task-checklist", 'value'),  
    Input('experiment-dropdown', 'value'),
    ])
def get_class_checklist(tasks, experiment_id):
    classes = db.get_task_classes(get_filter(experiment=experiment_id), tasks=tasks)
    return classes



@app.callback(Output('data-cache', 'children'),
              [
                Input('experiment-dropdown', 'value'),
                Input('interval-component', 'n_intervals')
                ],
                [State('data-cache', 'children')]
                )
def update_data_cache(experiment_id, n, cache_state):

    current_epoch = cache_state.get("epoch", -1)
    current_exp = cache_state.get("experiment_id")

    filter_by = get_filter(experiment=experiment_id)
    last_epoch = db.get_latest_epoch(filter_by)

    if experiment_id == current_exp:
        if last_epoch == current_epoch:
            return dash.no_update

    exp_config = db.experiments.find_one(filter={"experiment_id":experiment_id})

    filter_by["epoch"] =  { "$lte": last_epoch}
    scores = db.get_scores(filter_by)
    current_epoch = scores.epoch.max()
    outputs = db.outputs.find_one(filter={
                                    "experiment_id":experiment_id, 
                                    "epoch": int(current_epoch), 
                                    "split":"val"
                                    })

    if outputs:
        outputs.pop("_id")

    scores.pop("_id")
    exp_config.pop("_id")

    return {
            "exp_config": exp_config,
            "experiment_id": experiment_id,
            "epoch": current_epoch,
            "scores": scores.to_dict(),
            "outputs": outputs,
            }


@app.callback(
    Output('data-table', 'figure'),
    [Input('data-cache', 'children')]
    )
def get_data_table(data_cache):
    data = data_cache["exp_config"]["dataset_stats"]
    df = pd.DataFrame(data)
    return make_table(df, "Dataset Statistics")


@app.callback(
    Output('exp-config2', 'children'),
    [Input("config-dropdown2", "value"),
    Input('data-cache', 'children')]
    )
def get_config(config_value, data_cache):
    exp_config = data_cache["exp_config"]

    if config_value == "exp config":
        # exp_config.pop("trainer_args")
        # exp_config.pop("hyperparamaters")
        # exp_config.pop("_id")
        return json.dumps(exp_config["dataset_config"], indent=4)
    else:
        return json.dumps(exp_config[config_value], indent=4)



def loss_graph(data_cache, tasks):
    
    data = pd.DataFrame(data_cache["scores"])
    experiment_id = data_cache["experiment_id"]
    
    task_loss = [task+"-loss" for task in tasks]

    figure = make_lineplot(data, task_loss, "Loss")
    graph =  dcc.Graph(
                    id="loss-graph",
                    figure = fig_layout(figure)
                    )
    return graph


def task_metric_graph(data_cache, tasks, metrics):

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


def class_metric_graph(data_cache, tasks, metrics):

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


def highlight_text(data_cache):

    exp_config = data_cache["exp_config"]
    dataset = data_cache["exp_config"]["dataset"]
    sample_level = exp_config["dataset_config"]["sample_level"]
    epoch = data_cache["epoch"]

    sample_id = 6798

    gold_df = pd.DataFrame(db.datasets.find(filter={"dataset":dataset, f"{sample_level}_id":sample_id}))
    
    tokens = gold_df["text"].to_numpy()
    preds_dict = data_cache["outputs"]["data"]["preds"]["6798"]

    label_preds = ["_".join(x) for x in zip(preds_dict["seg"], preds_dict["ac"])]
    
    class_scores = {}

    # if "seg_ac" in output["preds"]:
    #     task = "seg_ac"
    # else:
    #     task = "seg"

    # label_preds = output["preds"][task]

    # if task == "seg_ac" and task in output["probs"]:
    #     pass
    # else:
    #     class_scores = {}

    #tokens = ["this", "is", "a", "claim", "sentence", "and", "this", "is", "a", "premise", "sentence","."]
    # class_scores = {
    #                 "claim":[0.2,0.8,0.9,0.9,0.7,0.4,0.3,0.1,0.0,0.1,0.2,0.1],
    #                 "premise":[0.1,0.1,0.2,0.2,0.1,0.1,0.2,0.0,0.7,0.8,0.7,0.1],
    #             }
    #preds = ["O","B-claim","I-claim","I-claim","I-claim","O","O","O","B-premise","I-premise","I-premise","O"]

    src =  mark_text( 
                        tokens,
                        class_scores, 
                        label_preds, 
                        [],
                        th_path
                    )

    img =  html.Img(   
                    id="text-highlight",
                    src=src
                    )

    return img


def tree_graph(data_cache):

    exp_config = data_cache["exp_config"]
    dataset = exp_config["dataset"]
    sample_level = exp_config["dataset_config"]["sample_level"]
    prediction_level = exp_config["dataset_config"]["prediction_level"]
    epoch = data_cache["epoch"]

    sample_id = 4
    sample_level = "document"

    gold_df = pd.DataFrame(db.datasets.find(filter={"dataset":dataset, f"{sample_level}_id":sample_id}))

    gold = {"ac": [], "relation": [], "text": []}

    #for i, group in grouped:
    acs = gold_df.groupby("ac_id")
    for i, ac in acs:
        gold["ac"].append(ac["ac"].unique()[0])
        gold["relation"].append(ac["relation"].unique()[0])
        gold["text"].append(" ".join(ac["text"].to_numpy()))


    gold["relation"] = [i + int(item) for i,item in enumerate(gold["relation"])]
    #preds = data_cache["outputs"]["data"][str(sample_id)]["preds"]
    #preds = {"ac": preds["ac"], "relation": preds["relation"]}
    #print(gold)
    #print(preds)

    print(gold)
    pred = gold

    figure  = make_tree_plot(gold, pred)
    graph =  dcc.Graph(
                    id="tree-graph",
                    figure = fig_layout(figure)
                    )
    return graph


def get_visual(key, cache, tasks, metrics):
    
    if key == "loss":
        return [loss_graph(cache, tasks)]
    elif key == "task_metric":
        return [task_metric_graph(cache, tasks, metrics)]
    elif key == "class_metric":
        return [class_metric_graph(cache, tasks, metrics)]
    elif key == "tree":
        return [tree_graph(cache)]
    elif key == "highlight":
        return [highlight_text(cache)]


@app.callback(Output('box1', 'children'),
              [
                Input('box1-dropdown', 'value'),
                Input('data-cache', 'children'),

                Input("task-checklist", 'value'),
                Input("metric-checklist", 'value')
                ])
def box1(visual, data_cache, tasks, metrics):
    return get_visual(visual, data_cache, tasks, metrics)


@app.callback(Output('box2', 'children'),
              [
                Input('box2-dropdown', 'value'),
                Input('data-cache', 'children'),

                Input("task-checklist", 'value'),
                Input("metric-checklist", 'value'),
                ])
def box1(visual, data_cache, tasks, metrics):
    return get_visual(visual, data_cache, tasks, metrics)


@app.callback(Output('box3', 'children'),
              [
                Input('box3-dropdown', 'value'),
                Input('data-cache', 'children'),

                Input("task-checklist", 'value'),
                Input("metric-checklist", 'value'),

                ])
def box1(visual, data_cache, tasks, metrics):
    return get_visual(visual, data_cache, tasks, metrics)


@app.callback(Output('box4', 'children'),
              [
                Input('box4-dropdown', 'value'),
                Input('data-cache', 'children'),

                Input("task-checklist", 'value'),
                Input("metric-checklist", 'value'),
                ])
def box1(visual, data_cache, tasks, metrics):
    return get_visual(visual, data_cache, tasks, metrics)
