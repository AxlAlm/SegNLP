

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
from pprint import pprint
import traceback
import logging

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
from .utils import get_filter, fig_layout, get_visible_info, update_visible_info
from .visuals import *


# from hotam import get_logger
# import inspect
# logger = get_logger("DASHBOARD", "-dashboard", logging_level=logging.DEBUG)

th_path  = "/tmp/text_highlight.png"


class Dashboard:

    def __init__(self, db):
        self.db = db
        app = dash.Dash("HotAm Dashboard")

        # app.logger  = logger
        # app.layout = html.Div([
        #                             dcc.Tabs(
        #                                     id='tabs', 
        #                                     value='tab-1', 
        #                                     children=[
        #                                                 dcc.Tab(label='Historical View', value='tab-1'),
        #                                                 dcc.Tab(label='Live View', value='tab-2'),
        #                                                 ]
        #                                     ),
        #                             html.Div(id='tab-content')
        #                         ])
        # app.callback(Output('tab-content', 'children'),
        #                     [Input('tabs', 'value')])(self.render_content)


        app.layout  = self.__get_content()



        #### RANKING DROPDOWNS / CHECKLISTS

        app.callback(Output('project-dropdown', 'options'),
                    Input('interval-component', 'n_intervals'))(self.update_project_dropdown)
        

        app.callback(Output('rank-filters', 'style'),
                    Input('project-dropdown', 'value'))(self.show_rank_filters)
        

        app.callback(Output('task-filter', 'options'),
                    Input('project-dropdown', 'value'))(self.update_task_dropdown)


        #### EXP VIEW DROPDOWN / CHECKLISTS
        app.callback(
                    Output('exp-dropdown', 'options'),
                    Output('exp-dropdown', 'value'),
                    [
                    Input('interval-component', 'n_intervals'),
                    Input('rank-graph', 'clickData')
                    ],
                    State('exp-dropdown', 'value'))(self.update_exp_dropdown)


        #### DATA CACHE 
        app.callback(
                    Output('data-cache', 'children'),
                    Output('exp-view', 'style'),
                    [
                    Input('interval-component', 'n_intervals'),
                    Input('exp-dropdown','value')
                    ],
                    [
                    State('data-cache', 'children'),
                    ])(self.update_data_cache)



        #### VISUALS
        app.callback(
                    Output('rank-graph', 'figure'),
                    Output('rank-view', 'style'),

                    Input('rank-btn-val', 'n_clicks'),

                    [
                    State('rank-graph', 'clickData'),
                    State('project-dropdown','value'),
                    State('task-filter', 'value'),
                    State('split-filter', 'value')
                    ]
                    )(self.update_rank_graph)

            
        app.callback(Output('exp-config', 'children'),
                    [Input("exp-config-dropdown", "value"),
                    Input('data-cache', 'children')])(self.get_config)


        app.callback(Output('data-dist-dropdown', 'options'),
                    Output('data-dist-dropdown', 'value'),
                    [Input('data-cache', 'children')],
                    [State('data-dist-dropdown', 'value')])(self.update_stats_dropdown)


        app.callback(Output('data-dist', 'figure'),
                    [Input('data-cache', 'children'),
                    Input('data-dist-dropdown', 'value')])(self.get_data_distributions)


        app.callback(
                    Output('loss-graph', 'figure'),
                    Output('loss-graph-con', 'style'),
                    [Input('data-cache', 'children')],
                    [State('loss-graph', 'figure')])(self.update_loss_graph)


        app.callback(
                    Output('task-metric-filter-task', 'options'),
                    Output('task-metric-filter-task', 'value'),
                    [Input('data-cache', 'children')],
                    [State('task-metric-filter-task','value')])(self.get_all_tasks)
        
        app.callback(
                    Output('task-metric-graph', 'figure'),
                    Output('task-metric-graph-con', 'style'),
                    [
                        Input('data-cache', 'children'),
                        Input('task-metric-filter-task', 'value'),
                        Input('task-metric-filter-metric', 'value'),
                        Input('task-metric-filter-split', 'value')
                    ],
                    [State('task-metric-graph', 'figure')])(self.update_task_metric_graph)


        app.callback(
                    Output('class-metric-filter-task', 'options'),
                    Output('class-metric-filter-task', 'value'),
                    [
                    Input('data-cache', 'children'),
                    ],
                    [State('class-metric-filter-task','value')])(self.get_all_tasks2)
        
        app.callback(
                    Output('class-metric-graph', 'figure'),
                    Output('class-metric-graph-con', 'style'),
                    [
                        Input('data-cache', 'children'),
                        Input('class-metric-filter-task', 'value'),
                        Input('class-metric-filter-metric', 'value'),
                        Input('class-metric-filter-split', 'value')
                    ],
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
    

        self.app = app


    def __get_info_div(self, name=""):
        if name:
            name = "-"+name
        info  = html.Div(
                                className="row flex-display",
                                children=[
                                            html.Div(  
                                                className="pretty_container six columns",
                                                children=[
                                                            dcc.Dropdown(
                                                                        id=f'exp-config-dropdown{name}',
                                                                        options=[
                                                                                    {"label": "hyperparamaters", "value":"hyperparamaters"},
                                                                                    {"label": "exp_config", "value":"exp_config"},
                                                                                    #{"label": "dataset_config", "value":"dataset_config"},
                                                                                    {"label": "trainer_args", "value":"trainer_args"}
                                                                                ],
                                                                        value="hyperparamaters",
                                                                        className="dcc_control",
                                                                        ),
                                                            html.Div( 
                                                                    children=[
                                                                                html.Pre(
                                                                                        id=f"exp-config{name}",
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
                                                                dcc.Dropdown(
                                                                            id=f'data-dist-dropdown{name}',
                                                                            options=[],
                                                                            value=None,
                                                                            className="dcc_control",
                                                                            ),
                                                                dcc.Graph(
                                                                    id=f"data-dist{name}",
                                                                    figure = go.Figure([])
                                                                    )
                                                            ]
                                                        )

                                        ]
                                    )
        return info
    

    def __get_visuals_div(self, name=""):

        if name:
            name = "-"+name

        visuals =  html.Div(
                            children=[
                                html.Div(
                                    className="row flex-display",
                                    #className="pretty_container twelve columns",
                                    id=f"loss-graph-con{name}",
                                    children=[
                                                html.Div(
                                                        #id=f"loss-graph-con{name}",
                                                        className="pretty_container twelve columns",
                                                        children=[dcc.Graph(
                                                                            id=f"loss-graph{name}",
                                                                            figure = go.Figure([])
                                                                            )
                                                                    ],
                                                        #style={'display': 'none'}
                                                        ),
                                                ],
                                    style={'display': 'none'}
                                    ),  
                                html.Div(
                                        id=f"task-metric-graph-con{name}",
                                        #className="pretty_container twelve columns",
                                        className="row flex-display",
                                        children=[
                                                html.Div(
                                                        #id="task-metric-filters",
                                                        className="pretty_container two columns",
                                                        children=[
                                                                    html.Div( 
                                                                            #className="pretty_container six columns",
                                                                            #className="row",
                                                                            children=[
                                                                                    html.P("Filter Task:", className="control_label"),
                                                                                    dcc.Checklist(
                                                                                                id=f"task-metric-filter-task{name}",
                                                                                                options=[],
                                                                                                value=None,
                                                                                                className="dcc_control",
                                                                                                labelStyle={'display': 'inline-block'}
                                                                                                ),
                                                                                        ]
                                                                                ),
                                                                        html.Div( 
                                                                            #className="pretty_container six columns",
                                                                            #className="column flex-display",
                                                                            children=[
                                                                                    html.P("Filter Metric:", className="control_label"),
                                                                                    dcc.Checklist(
                                                                                                id=f"task-metric-filter-metric{name}",
                                                                                                options=[
                                                                                                        {"label":"f1", "value":"f1"},
                                                                                                        {"label":"precision", "value":"precision"},
                                                                                                        {"label":"recall", "value":"recall"}
                                                                                                        ],
                                                                                                value=["f1"],
                                                                                                className="dcc_control",
                                                                                                labelStyle={'display': 'inline-block'}
                                                                                                ),
                                                                                        ]
                                                                                ),
                                                                    html.Div( 
                                                                            #className="pretty_container six columns",
                                                                            #className="column flex-display",
                                                                            children=[
                                                                                        html.P("Filter Split:", className="control_label"),
                                                                                        dcc.Checklist(
                                                                                                id=f"task-metric-filter-split{name}",
                                                                                                options=[
                                                                                                        {"label":"val", "value":"val"},
                                                                                                        {"label":"train", "value":"train"}
                                                                                                        ],
                                                                                                value=["val", "train"],
                                                                                                className="dcc_control",
                                                                                                labelStyle={'display': 'inline-block'}
                                                                                                ),    
                                                                                        ]
                                                                                ),
                                                                        ],
                                                        ),
                                                html.Div(
                                                        className="pretty_container nine columns",
                                                        children=[
                                                                dcc.Graph(
                                                                    id=f"task-metric-graph{name}",
                                                                    figure = go.Figure([])
                                                                    )
                                                                    ]
                                                        )
                                            ],                                                    
                                        style={'display': 'none'}
                                ),
                                html.Div(
                                        id=f"class-metric-graph-con{name}",
                                        #className="pretty_container twelve columns",
                                        className="row flex-display",
                                        children=[
                                                html.Div(
                                                        #id="task-metric-filters",
                                                        className="pretty_container two columns",
                                                        children=[
                                                                    html.Div( 
                                                                            #className="pretty_container six columns",
                                                                            #className="row",
                                                                            children=[
                                                                                    html.P("Filter Task:", className="control_label"),
                                                                                    dcc.Dropdown(
                                                                                                id=f"class-metric-filter-task{name}",
                                                                                                options=[],
                                                                                                value=None,
                                                                                                className="dcc_control",
                                                                                                ),
                                                                                        ]
                                                                                ),
                                                                        html.Div( 
                                                                            #className="pretty_container six columns",
                                                                            #className="column flex-display",
                                                                            children=[
                                                                                    html.P("Filter Metric:", className="control_label"),
                                                                                    dcc.Checklist(
                                                                                                id=f"class-metric-filter-metric{name}",
                                                                                                options=[
                                                                                                        {"label":"f1", "value":"f1"},
                                                                                                        {"label":"precision", "value":"precision"},
                                                                                                        {"label":"recall", "value":"recall"}
                                                                                                        ],
                                                                                                value=["f1"],
                                                                                                className="dcc_control",
                                                                                                labelStyle={'display': 'inline-block'}
                                                                                                ),
                                                                                        ]
                                                                                ),
                                                                    html.Div( 
                                                                            #className="pretty_container six columns",
                                                                            #className="column flex-display",
                                                                            children=[
                                                                                        html.P("Filter Split:", className="control_label"),
                                                                                        dcc.Checklist(
                                                                                                id=f"class-metric-filter-split{name}",
                                                                                                options=[
                                                                                                        {"label":"val", "value":"val"},
                                                                                                        {"label":"train", "value":"train"}
                                                                                                        ],
                                                                                                value=["val", "train"],
                                                                                                className="dcc_control",
                                                                                                labelStyle={'display': 'inline-block'}
                                                                                                ),    
                                                                                        ]
                                                                                ),
                                                                        ],
                                                        ),
                                                html.Div(
                                                        className="pretty_container nine columns",
                                                        children=[
                                                                dcc.Graph(
                                                                    id=f"class-metric-graph{name}",
                                                                    figure = go.Figure([])
                                                                    )
                                                                    ]
                                                        )
                                            ],                                                    
                                        style={'display': 'none'}
                                ),
                                html.Div(
                                    className="row flex-display",
                                    children=[
                                                # html.Div(
                                                #         id=f"class-metric-graph-con{name}",
                                                #         className="pretty_container six columns",
                                                #         children=[dcc.Graph(
                                                #                             id=f"class-metric-graph{name}",
                                                #                             figure = go.Figure([])
                                                #                             )
                                                #                             ],                                                      
                                                #         style={'display': 'none'}
                                                #         ),
                                                html.Div(
                                                        id=f"conf-matrix-con{name}",
                                                        className="pretty_container six columns",
                                                        children=[
                                                                    dcc.Dropdown(
                                                                                id=f'conf-dropdown{name}',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                ),
                                                                    dcc.Graph(
                                                                                id=f"conf-matrix{name}",
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
                                                        id=f"tree-graph-con{name}",
                                                        className="pretty_container twelve columns",
                                                        children=[
                                                                    dcc.Dropdown(
                                                                                id=f'sample-id-dropdown2{name}',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                ),
                                                                    dcc.Graph(
                                                                            id=f"tree-graph{name}",
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
                                                        id=f"text-highlight-con{name}",
                                                        className="pretty_container twelve columns",
                                                        children=[ 
                                                                    dcc.Dropdown(
                                                                                id=f'sample-id-dropdown1{name}',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                ),
                                                                    html.Img(   
                                                                            id=f"text-highlight{name}",
                                                                            src=""
                                                                            ),
                                                                    ],
                                                        style={'display': 'none'}
                                                        ),
                                                ]
                                        )
                                ]
                            )
        return visuals


    def __get_content(self):
        return html.Div(    
                        children=[  

                                html.Div(
                                        className="column flex-display",
                                        children=[
                                                html.Div(
                                                        className="pretty_container two columns",
                                                        children=[
                                                                    html.P("Choose a Project:", className="control_label"),
                                                                    dcc.Dropdown(
                                                                                id='project-dropdown',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                placeholder="Select a project",
                                                                                #persistence=True,
                                                                                #clearable=False,
                                                                                ),
                                                                    html.Div(
                                                                            id="rank-filters",
                                                                            className="row",
                                                                            children=[
                                                                                        html.Div( 
                                                                                                #className="pretty_container six columns",
                                                                                                #className="column flex-display",
                                                                                                children=[
                                                                                                        html.P("Rank by task:", className="control_label"),
                                                                                                        dcc.RadioItems(
                                                                                                                    id='task-filter',
                                                                                                                    options=[],
                                                                                                                    value=None,
                                                                                                                    className="dcc_control",
                                                                                                                    labelStyle={'display': 'inline-block'}
                                                                                                                    ),
                                                                                                            ]
                                                                                                    ),
                                                                                        html.Div( 
                                                                                                #className="pretty_container six columns",
                                                                                                #className="column flex-display",
                                                                                                children=[
                                                                                                            html.P("Rank by split:", className="control_label"),
                                                                                                            dcc.RadioItems(
                                                                                                                    id='split-filter',
                                                                                                                    options=[
                                                                                                                            {"label":"val", "value":"val"},
                                                                                                                            {"label":"test", "value":"test"}
                                                                                                                            ],
                                                                                                                    value="val",
                                                                                                                    className="dcc_control",
                                                                                                                    labelStyle={'display': 'inline-block'}
                                                                                                                    ),    
                                                                                                            ]
                                                                                                    ),
                                                                                            html.Button('Rank', id='rank-btn-val', n_clicks=0),
                                                                                            ],
                                                                            style={'display': 'none'}
                                                                            ),
                                                                    ]
                                                                ),
                                                html.Div(   
                                                        id="rank-view",
                                                        className="pretty_container ten columns",
                                                        children=[
                                                                    dcc.Graph(
                                                                                id="rank-graph",
                                                                                figure = go.Figure([])
                                                                                ),                                        
                                                                ],
                                                        style={'display': 'none'}
                                                        ),
                                                    ]
                                        ),
                                    html.Div(
                                            className="pretty_container twelve columns",
                                            children=[
                                                        dcc.Dropdown(
                                                                    id='exp-dropdown',
                                                                    options=[],
                                                                    value=None,
                                                                    className="dcc_control",
                                                                    placeholder="Select a Experiment ID",
                                                                    #persistence=True,
                                                                    #clearable=False,
                                                                    ),
                                                        ]
                                            ),
                                    html.Div(   
                                            id="exp-view",
                                            className="column",
                                            children=[
                                                        self.__get_info_div(),
                                                        self.__get_visuals_div()
                                                        ],
                                            style={'display': 'none'}
                                            #style={"display": "flex", "flex-direction": "column"},
                                            ),
                                    dcc.Interval(
                                                    id='interval-component',
                                                    interval=1*1000, # in milliseconds
                                                    n_intervals=0,
                                                    max_intervals=-1,
                                                                                ),
                                    html.Div(id='data-cache', children=dict(), style={'display': 'none'}),
                                    ]
                        )


    def update_project_dropdown(self, n):
        projects = self.db.get_projects()
        return [{"label":p, "value":p} for p in projects]


    # def update_live_exp_dropdown(self, n):
    #     exps = sorted(self.db.get_live_exps_ids())
    #     return [{"label":e, "value":e} for e in exps]


    def update_exp_dropdown(self, n, clickData, value):

        if clickData:
            value = dict(clickData)["points"][0]["customdata"]

        # live_exps = set(self.db.get_live_exps_ids())
        # done_exps = set(self.db.get_done_exps_ids())

        exps = self.db.get_exp_ids()

        #exps = sorted(live_exps | done_exps)
        options = [{"label":e, "value":e} for e  in exps]

        return options, value

    
    def update_task_dropdown(self, project):
        tasks = sorted(self.db.get_project_tasks(project))
        return [{"label":t, "value":t} for t in tasks]


    def get_all_tasks(self, data_cache, value):
        
        if not data_cache:
            return [], None

        all_tasks = sorted(set(data_cache["exp_config"]["tasks"] + data_cache["exp_config"]["subtasks"]))
        options = [{"label":t, "value":t} for t in all_tasks]

        if value is None:
            value = data_cache["exp_config"]["tasks"]

        return options, value


    def get_all_tasks2(self, data_cache, value):
        """
        we filter tasks that are unions for relation + something. Relation itself is fine, but not in combination
        """
        
        if not data_cache:
            return [], None

        all_tasks = sorted(set(data_cache["exp_config"]["tasks"] + data_cache["exp_config"]["subtasks"]))
        options = [{"label":t, "value":t} for t in all_tasks if ("relation" not in t or t == "relation")]

        if value is None:
            value = options[0]["value"]

        return options, value


    def update_stats_dropdown(self, data_cache, value):
        exp_config = data_cache.get("exp_config", {})

        if not exp_config:
            return [], None
            
        subtasks = exp_config["subtasks"]
        tasks = exp_config["tasks"]
        all_tasks = sorted(set(subtasks + tasks))
        options = [{"label":t, "value":t} for t in all_tasks]

        if value is None:
            value = all_tasks[0]

        return options, value


    def get_data_distributions(self, data_cache, value):
        
        #print("DATA CHACE", data_cache)
        if not data_cache:
            return go.Figure()

        if not value:
            return go.Figure()

        data = data_cache["exp_config"]["dataset_stats"]
        df = pd.DataFrame(data)


        df = df[df["task"] == value]

        if value == "relation":
            return make_relation_dist_plot(df)
        else:
            return label_dist_plot(df)


    def get_config(self, config_value, data_cache):
        exp_config = data_cache.get("exp_config", {})

        if config_value == "exp_config":
            exp_config.pop("trainer_args")
            exp_config.pop("hyperparamaters")
            #exp_config.pop("dataset_config")
            #exp_config.pop("_id")
            return json.dumps(exp_config, indent=4)
        else:
            return json.dumps(exp_config.get(config_value,{}), indent=4)


    def update_loss_graph(self, data_cache, fig_state):

        fig_state = go.Figure(fig_state)

        if not data_cache:
            return fig_state, {"display":"none"}

        if not data_cache["scores"]:
            return fig_state, {"display":"none"}


        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]
        tasks = data_cache["exp_config"]["tasks"]
        
        task_loss = [task+"-loss" for task in tasks]

        
        filtered_data = data.loc[:, task_loss+ ["split", "epoch"]]
        
        max_loss = data.loc[:, task_loss].max().max()
        max_loss = max_loss + (max_loss*0.2)
        figure = make_lineplot(filtered_data, "Loss", max_y=max_loss)

        #figure = make_lineplot(data, task_loss, "Loss")

        last_vis_state = get_visible_info(fig_state)
        current_vis_state = get_visible_info(figure)
        if last_vis_state.keys() == current_vis_state.keys():
            update_visible_info(figure, last_vis_state)

        return figure, {"display":"block"}


    def update_task_metric_graph(self, data_cache, tasks, metrics, splits, fig_state):


        fig_state = go.Figure(fig_state)

        if not data_cache:
            return fig_state, {"display":"none"}

        if not data_cache["scores"]:
            return fig_state, {"display":"none"}

        # if task is None:
        #     return fig_state, {"display":"none"}

        data = pd.DataFrame(data_cache["scores"])
        experiment_id = data_cache["experiment_id"]
        
        #tasks = data_cache["exp_config"]["tasks"] + data_cache["exp_config"]["subtasks"]
        #metrics = data_cache["exp_config"]["metrics"].copy()
        #metrics.remove("confusion_matrix")       

        task_metrics = []
        for task in tasks:
            for metric in metrics:
                task_metrics.append("-".join([task,metric]))
    
        cond = data["split"].isin(splits) 
        filtered_data = data.loc[cond].loc[:, task_metrics+ ["split", "epoch"]]

        figure = make_lineplot(filtered_data, "Task Scores", max_y=1.0)

        last_vis_state = get_visible_info(fig_state)
        if data_cache["epoch"] == 0:
            last_vis_state = {k:False for k in task_metrics if "f1" not in k}

        current_vis_state = get_visible_info(figure)
        if last_vis_state.keys() == current_vis_state.keys():
            update_visible_info(figure, last_vis_state)

        return figure, {"display":"block"}


    def update_class_metric_graph(self, data_cache, task, metrics, splits, fig_state):

        fig_state = go.Figure(fig_state)

        if not data_cache:
            return fig_state, {"display":"none"}

        if not data_cache["scores"]:
            return fig_state, {"display":"none"}


        data = pd.DataFrame(data_cache["scores"])
        task2labels = data_cache["exp_config"]["dataset_config"]["task_labels"]
        #experiment_id = data_cache["experiment_id"]

        #tasks = data_cache["exp_config"]["tasks"]
        # if len(tasks) == 1 and tasks[0] == "relation":
        #     return fig_state, {"display":"none"}
        # metrics = data_cache["exp_config"]["metrics"].copy()
        # metrics.remove("confusion_matrix")

        if task == "relation":

            filter_columns = []
            for c in task2labels[task]:
                filter_columns.append("-".join([task, str(c), "f1"]).lower())

            data_stats = data_cache["exp_config"]["dataset_stats"]
            figure = make_rel_error_dist_plot(data, data_stats, splits, filter_columns)

        else:
        
            filter_columns = []
            for c in task2labels[task]:
                for metric in metrics:
                    filter_columns.append("-".join([task, str(c), metric]).lower())


            cond = data["split"].isin(splits) 
            filtered_data = data.loc[cond].loc[:, filter_columns+ ["split", "epoch"]]

            figure = make_lineplot(filtered_data, "Class Scores", max_y=1.0)

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
        return "", {'display': 'none'}

        if not data_cache or sample_id is None:
            return "", {'display': 'none'}

        if not data_cache["outputs"]:
            return "", {'display': 'none'}

        tasks  = set(data_cache["exp_config"]["tasks"] +data_cache["exp_config"]["subtasks"])
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
            pred_span = sample_out["preds"]["span_ids"][i] if "SPAN" in sample_out["preds"]["span_ids"][i] else None
            gold_span = sample_out["gold"]["span_ids"][i] if "SPAN" in sample_out["gold"]["span_ids"][i] else None

            token_data = {
                            "token": token,
                            "pred":{
                                    "span_id": pred_span,
                                        }
                            }

            if pred_span:
                pred_label = sample_out["preds"][task][i]
                pred_label = pred_label if pred_label != "None" else None
                token_data["pred"]["label"] = pred_label

            if task in sample_out["probs"]:
                scores = sample_out["probs"][task][i]
                idx = np.argmax(scores)
                token_data["pred"]["score"] = scores[idx]
                token_data["pred"]["label"] = task2labels[task][idx]

            if gold_span:
                gold_label = sample_out["gold"][task][i]
                gold_label = gold_label if gold_label != "None" else None
                token_data["gold"] ={
                                    "span_id":gold_span,
                                    "label":gold_label
                                    }


            data.append(token_data)

        #pprint(data)
        labels = [l for l in task2labels[task] if l != "None"]
        hotviz.hot_text(data, labels=labels, save_path=th_path, print_html=False, width=1200)


        with open(th_path, "rb") as f:
            enc_img = base64.b64encode(f.read())
            src = f"data:image/png;base64,{enc_img.decode()}"

        return src, {'display': 'block'}


    def update_tree_graph(self, sample_id, data_cache, fig_state):

        def extract_data(df):
            spans = df.groupby("span_ids")
            span_data = []
            for i,(span_id, span) in enumerate(spans):

                label = span["ac"].unique()[0]
                relation = span["relation"].unique()[0]
                relation_type = span["stance"].unique()[0]
                span_data.append({
                                    "label":label,
                                    "link": relation,
                                    "link_label": relation_type,
                                    "text": " ".join(span["text"].tolist()),
                                    })            
            return span_data


        def pe_majorclaim_fix(data):
            """
            In persuasive essays dataset all Claims are related to the majorlcaims. the Major claims are parapharses of the same statement.
            """
            majorclaim_idx = None
            for i,d in enumerate(data):

                if d["label"] == "MajorClaim":
                    if majorclaim_idx is not None:
                        d["link"] = majorclaim_idx
                        d["link_label"] = "paraphrase"
                    else:
                        majorclaim_idx = i

            for d in data:
                if d["label"] == "Claim":
                    d["link"] = majorclaim_idx


        fig_state = go.Figure(fig_state)

        if not data_cache or sample_id is None:
            return fig_state, {'display': 'none'}

        if not data_cache["scores"]:
            return fig_state, {"display":"none"}

        all_tasks  = set(data_cache["exp_config"]["tasks"] + data_cache["exp_config"]["subtasks"])
        if "relation" not in all_tasks:
            return go.Figure([]), {'display': 'none'}

        has_stance = False
        if "stance" in all_tasks:
            has_stance = True

        outputs = data_cache["outputs"]

        if not outputs:
            return fig_state, {"display":"none"}
        
        sample_out = outputs[sample_id]

        pred_df = pd.DataFrame(sample_out["preds"])
        gold_df = pd.DataFrame(sample_out["gold"])

        if "span_ids" in pred_df.columns:
            text = sample_out["text"]
            pred_df["text"] = text
            gold_df["text"] = text
            pred_data = extract_data(pred_df)
            gold_data = extract_data(gold_df)

      
        else:
            text = [" ".join(t) for t in sample_out["text"]]

            rename_dict = {"ac": "label", "relation": "link", "stance":"link_label"}

            if not has_stance:
                pred_df["stance"] = "None"
                gold_df["stance"] = "None"

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

        # gold_data = [d for d in gold_data if d["label"] != "None"]
        # pred_data = [d for d in pred_data if d["label"] != "None"]

        # gold_data = [d for d in gold_data if d["link"] < len(gold_data)]
        # pred_data = [d for d in pred_data if d["link"] < len(pred_data)]

        if data_cache["exp_config"]["dataset"] == "pe":
            pe_majorclaim_fix(pred_data)
            pe_majorclaim_fix(gold_data)

        
        #pprint(pred_data)
        # pprint(gold_data)

        print("WTFF IS HAPPING")
        #try:
        fig = hotviz.hot_tree(pred_data, gold_data=gold_data) # link_label2color={"For":"#28cb44", "supports":"#28cb44","Against":"#f21535", "attacks":"#f21535", "Paraphrase":"#f58a00"})
        #print(fig)
        #print("AER WE DONE")
        fig.update_layout(
                            autosize=False,
                            width=1200,
                            height=1200,
                        )


        last_vis_state = get_visible_info(fig_state)
        current_vis_state = get_visible_info(fig)
        if last_vis_state.keys() == current_vis_state.keys():
            update_visible_info(fig, last_vis_state)
        
        return fig, {'display': 'block'}


    def update_conf_dropdown(self, experiment_id):
        exp_config = self.db.get_exp_config({"experiment_id":experiment_id})

        if exp_config is None:
            return [], None

        options = [{"label":t, "value":t} for t in exp_config["tasks"] if "relation" not in t]

        if not options:
            value = None
        else:
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

        conf_m_probs = []
        for i in range(conf_m.shape[0]):
            #print(conf_m[i], np.sum(conf_m[i]), np.round(conf_m[i] / np.sum(conf_m[i])))
            conf_m_probs.append(list(np.round_(conf_m[i] / np.sum(conf_m[i]), decimals=2)))

        labels = data_cache["exp_config"]["dataset_config"]["task_labels"][task]
        fig = conf_matrix(conf_m_probs, labels)

        return fig, {'display': 'block'}


    def update_rank_graph(self, click_n, clickData, project, task, rank_split):


        if None in [project, task, rank_split]:
            return go.Figure([]), {"display":"none"}


        rank_metric = "f1"
        task = task + "-" if task != "mean" else ""
        top_n = 10
        #dataset_name, project, rank_task, rank_metric, rank_split, top_n = rank_values

        filter_by = get_filter(project=project)
        experiments = pd.DataFrame(self.db.get_exp_configs(filter_by))
        experiment_ids = list(experiments["experiment_id"].to_numpy())

        data = self.db.get_scores(experiment_ids=experiment_ids)


        experiment2config = {}
        for i, exp_row in experiments.iterrows():
            exp_id = exp_row["experiment_id"]
            config = exp_row.to_dict()
            config.pop("_id")
            experiment2config[exp_id] = config
        
        #NOTE! we are assuming that all experiments are done with the same metrics
        # and we are displaying only
        display_splits = ["val", "test"] 
        display_metrics = [m for m in list(experiment2config.items())[-1][1]["metrics"] if "confusion" not in m]
        
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
                task_metric = f"{task}{metric}"

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
        score_df.sort_values(f"{rank_split}-{task}{rank_metric}", inplace=True, ascending=ascend)

        top_scores = score_df.head(top_n)
        top_scores.reset_index(drop=True, inplace=True)

        fig = rank_bar( 
                        title=f"Top {top_n} experiments for project {project}",
                        top_scores=top_scores, 
                        display_metrics=[rank_metric], 
                        display_splits=[rank_split], 
                        experiment2config=experiment2config, 
                        rank_task=task, 
                        top_n=top_n, 
                        clickdata=clickData
                        )

        return fig, {'display': 'block'}


    def show_rank_filters(self, value):
        if value is not None:
            return {'display': 'block'}
        else:
            return {'display': 'none'}


    def get_exp_data(self, experiment_id:str, last_epoch:int):

    
        exp_config = self.db.get_exp_config(experiment_id=experiment_id)

        if exp_config is None:
            return {}, {"display":"none"}

        scores = pd.DataFrame(self.db.get_scores(experiment_id=experiment_id, epoch=last_epoch))
        scores = scores.to_dict()

        outputs = self.db.get_outputs(experiment_id=experiment_id, epoch=last_epoch).get("data", {})
        
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


    def update_data_cache(self, n, experiment_id,  cache_state):

        if experiment_id is None:
            return dash.no_update

        current_exp = cache_state.get("experiment_id", None)
        prev_epoch = cache_state.get("epoch", -1)
        status = cache_state.get("exp_config", {}).get("status", "done")
        #if cache_state != {}:

        if current_exp == experiment_id and status == "done":
            return dash.no_update

        last_epoch = self.db.get_last_epoch(experiment_id=experiment_id)

        # if there is now new epoch to update for
        if last_epoch == prev_epoch:
            return dash.no_update

        data_cache, style  = self.get_exp_data(experiment_id, last_epoch)
        return data_cache, style


    def run_server(self, *args, **kwargs):
        self.app.run_server(*args, **kwargs)


