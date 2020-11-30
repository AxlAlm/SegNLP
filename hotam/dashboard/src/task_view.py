
#basics
import pandas as pd
import re
import numpy as np
import json

#dash
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State

#plotly
import plotly
import plotly.express as px
import plotly.graph_objs as go

#import app and mongodb
from src.servers import app, db

from src.utils import get_filter, dropdown_filter, fig_layout
from src.visuals import *


filters  = html.Div(  
                        className="pretty_container four columns",
                        children=[
                                    html.H1('Filters'),
                                    html.P("Filter by Dataset:", className="control_label"),
                                    dcc.Dropdown(
                                                id='dataset-dropdown2',
                                                options=db.get_dropdown_options("dataset"),
                                                value=db.get_last_exp().get("dataset",[]),
                                                className="dcc_control",

                                            ),
                                    html.P("Filter by Project:", className="control_label"),
                                    dcc.Dropdown(
                                                id='project-dropdown2',
                                                options=db.get_dropdown_options("project"),
                                                value=db.get_last_exp().get("project",[]),
                                                className="dcc_control",
                                            ), 
                                    html.P("Filter by Task:", className="control_label"),
                                    dcc.Dropdown(
                                                id='task-rank-dropdown',
                                                options=[],
                                                value="seg",
                                                className="dcc_control",
                                            ), 
                                    html.P("Filter by Metric:", className="control_label"),
                                    dcc.Dropdown(
                                                id='metric-rank-dropdown',
                                                options=[],
                                                value="f1",
                                                className="dcc_control",
                                            ), 
                                    html.P("Rank by Split:", className="control_label"),
                                    dcc.Dropdown(
                                                id='split-rank-dropdown',
                                                options=[
                                                        {"label":"val", "value":"val"},
                                                        {"label":"test", "value":"test"}
                                                        ],
                                                value="val",
                                                className="dcc_control",
                                            ), 
                                    html.P("Top N:", className="control_label"),
                                    dcc.Dropdown(
                                                id='top-rank-dropdown',
                                                options=[{"label":i,"value":i} for i in range(1,11)],
                                                value=5,
                                                className="dcc_control",
                                                ),
                                    html.Button('Submit', id='rank-button', n_clicks=0),
                                    html.Div(   
                                            id='rank-button-container',
                                            style={'display': 'none'},
                                            children=[]
                                            )
                                    ]
                        )

plot_filter = html.Div(  
                            className="pretty_container six columns",
                            children=[  

                                        html.Div(id='rank-cache', style={'display': 'none'}),
                                        html.H3('Display Option'),
                                        dcc.Checklist(
                                            id="split-checklist",
                                            options=[
                                                    {"label":"val", "value":"val"},
                                                    {"label":"test", "value":"test"}
                                                    ],
                                            value=["val"],
                                            labelStyle={'display': 'inline-block'},
                                            className="dcc_control",
                                        ),
                                        dcc.Checklist(
                                            id="metric-checklist2",
                                            options=[],
                                            value=db.get_last_exp().get("metrics",[]),
                                            labelStyle={'display': 'inline-block'},
                                            className="dcc_control",
                                        ),
                                    ]
                            )

rank_plot = html.Div(
                    className="pretty_container ten columns",
                    children=[dcc.Graph(
                                id='top_bar', 
                                clickData={},
                                figure = fig_layout(go.Figure(data=[]))
                                )]
                    )


plot_filter_row = html.Div(
                                className="row",
                                children=[  
                                            plot_filter,
                                            ],
                            )

filter_rank_plot = html.Div(
                            className="row flex-display",
                            children=[  
                                        filters,
                                        plot_filter_row,
                                        rank_plot
                                        ]
                        )


exp_plot = html.Div(
                    className="pretty_container ten columns",
                    children=[dcc.Graph(
                                        id='exp_plot', 
                                        clickData={},
                                        figure = fig_layout(go.Figure(data=[]))
                                        )],
                    )


exp_details = html.Div(  
                        className="pretty_container five columns",
                        children=[
                                    dcc.Dropdown(
                                                id='config-dropdown',
                                                options=[
                                                            {"label": "hyperparamaters", "value":"hyperparamaters"},
                                                            {"label": "project setting", "value":"project setting"},
                                                            {"label": "trainer_args", "value":"trainer_args"}
                                                        ],
                                                value="hyperparamaters",
                                                className="dcc_control",
                                                ),
                                    html.Div( 
                                            className="pretty_container",
                                            children=[
                                                        html.Pre(
                                                                id="exp-config",
                                                                children=""
                                                                )
                                                        ]
                                            )
                                    ]       
                        )

exp_view =   html.Div(  
                            className="row flex-display",
                            children = [
                                        exp_details,
                                        exp_plot

                                            ]
                        )
                                    
    

overview_sides = html.Div(  
                            #className="row flex-display",  # Define the row element
                            children=[
                                        filter_rank_plot,
                                        exp_view
                                    ],
                            style={"display": "flex", "flex-direction": "column"},
                        )



@app.callback(
    Output('metric-rank-dropdown', 'options'),
    [Input('dataset-dropdown2', 'value'),
    Input('project-dropdown2', 'value')])
def update_metric_rank_dropdown(dataset_name, project):
    return db.get_metrics(get_filter(dataset=dataset_name, project=project))


@app.callback(
    Output('task-rank-dropdown', 'options'),
    [Input('dataset-dropdown2', 'value'),
    Input('project-dropdown2', 'value')])
def update_task_rank_dropdown(dataset_name, project):
    return db.get_tasks(get_filter(dataset=dataset_name, project=project))


@app.callback(
    Output("metric-checklist2", 'options'),
    [Input('dataset-dropdown2', 'value'),
    Input('project-dropdown2', 'value')])
def get_metric_checklist2(dataset_name, project):
    return db.get_metrics(get_filter(dataset=dataset_name, project=project))


@app.callback(Output('rank-button-container', 'children'),
              [ 
                Input("dataset-dropdown2", 'value'),
                Input("project-dropdown2", 'value'),
                Input("task-rank-dropdown", "value"),
                Input("metric-rank-dropdown", "value"),
                Input("split-rank-dropdown", "value"),
                Input("top-rank-dropdown", "value"),
                ])
def rank_submit(dataset_name, project, rank_task, rank_metric, rank_split, top_n):
    return [dataset_name, project, rank_task, rank_metric, rank_split, top_n]


@app.callback(Output('rank-cache', 'children'),
              [ Input('rank-button',"n_clicks"),
                State('rank-button-container', 'children')])
def rank_data(n_clicks, rank_values):

    if not rank_values:
        return {}

    dataset_name, project, rank_task, rank_metric, rank_split, top_n = rank_values

    filter_by = get_filter(dataset=dataset_name, project=project)
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
            task_metric = f"{rank_task}-{metric}"

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
    score_df.sort_values(f"{rank_split}-{rank_task}-{rank_metric}", inplace=True, ascending=ascend)

    top_scores = score_df.head(top_n)
    top_scores.reset_index(drop=True, inplace=True)

    data_cache = {}
    data_cache["top_scores"] = top_scores.to_dict()
    data_cache["experiment2config"] = experiment2config
    data_cache["rank_task"] = rank_task

    return data_cache



@app.callback(Output('top_bar', 'figure'),
              [ 
                Input('top_bar', 'clickData'),
                Input("metric-checklist2", "value"),
                Input("split-checklist",'value'),
                Input('rank-cache', "children"),
                #Input('interval-component2', 'n_intervals')
                ])
def update_top_bar(clickdata, display_metrics, display_splits, data_cache):
    
    if "top_scores" not in data_cache:
        return go.Figure(data=[])

    top_scores = pd.DataFrame(data_cache["top_scores"])
    experiment2config = data_cache["experiment2config"]
    rank_task = data_cache["rank_task"]
    top_n = top_scores.shape[0]

    return rank_bar(top_scores, display_metrics, display_splits, experiment2config, rank_task, top_n, clickdata)


@app.callback(
    Output('exp_plot', 'figure'),
    [Input('top_bar', 'clickData')])
def display_click_data(clickData):

    if not clickData:
        fig = go.Figure(data=[])
    else:

        exp_id = dict(clickData)["points"][0]["customdata"]
        filter_by = get_filter(experiment=exp_id)
        exp_config = db.experiments.find_one(filter_by)
        exp_config.pop("_id")
        hyperparams = exp_config.pop("hyperparamaters")
        trainer_args = exp_config.pop("trainer_args")

        exp_data = db.get_scores(filter_by)
        grouped = exp_data.groupby("split")

        # Create the graph with subplots
        fig = plotly.subplots.make_subplots(    
                                                rows=2, 
                                                cols=1, 
                                                shared_xaxes=False,
                                                #subplot_titles=("",),
                                                vertical_spacing=0.2,
                                                )
        

        fig.add_trace({
            'x': ["class1", "class2"],
            'y': [2,3],
            'name': f"Class Scores",
            #'mode': 'lines+markers',
            'type': 'bar',
            #"connectgaps":True
            }, 1, 1)

        
    return fig_layout(fig)


@app.callback(
    Output('exp-config', 'children'),
    [Input('top_bar', 'clickData'),
    Input('config-dropdown', 'value')])
def show_config(clickData, config_value):

    if not clickData:
        return []

    exp_id = dict(clickData)["points"][0]["customdata"]
    filter_by = get_filter(experiment=exp_id)
    exp_config = db.experiments.find_one(filter_by)

    if config_value == "project setting":
        exp_config.pop("trainer_args")
        exp_config.pop("hyperparamaters")
        return json.dumps(exp_config, indent=4)
    else:
        return json.dumps(exp_config[config_value], indent=4)

