
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


from .base import ViewBase
from hotam.dashboard.utils import get_filter, fig_layout, get_visible_info, update_visible_info
from hotam.dashboard.views.visuals import *


th_path  = "/tmp/text_highlight.png"

class HistView(ViewBase):

    def __init__(self, app, db):
        self.db = db

        info = html.Div(
                        className="row flex-display",
                        children=[
                                    html.Div(  
                                        className="pretty_container six columns",
                                        children=[
                                                    dcc.Dropdown(
                                                                id='exp-config-dropdown-hist',
                                                                options=[
                                                                            {"label": "hyperparamaters", "value":"hyperparamaters"},
                                                                            {"label": "exp config", "value":"exp config"},
                                                                            {"label": "dataset_config", "value":"dataset_config"},
                                                                            {"label": "trainer_args", "value":"trainer_args"}
                                                                        ],
                                                                value="hyperparamaters",
                                                                className="dcc_control",
                                                                ),
                                                    html.Div( 
                                                            className="pretty_container six columns",
                                                            children=[
                                                                        html.Pre(
                                                                                id="exp-config-hist",
                                                                                children=""
                                                                                )
                                                                        ],
                                                            style={
                                                                    "maxHeight": "500px", 
                                                                    "maxWidth": "300px",
                                                                    "overflow": "scroll",
                                                                    }
                                                            )
                                                    ],
                                                style={'display': 'none'}
                                        ),
                                    html.Div( 
                                            className="pretty_container six columns",
                                            children=[
                                                        dcc.Graph(
                                                                    id='data-table-hist',
                                                                    figure = go.Figure(data=[]),
                                                            )
                                                        
                                                    ],
                                            style={'display': 'none'}
                                            )

                                ]
                            )
    

        
        visuals =  html.Div(
                            children=[
                                html.Div(
                                    className="row flex-display",
                                    children=[
                                                html.Div(
                                                        id="loss-graph-con-hist",
                                                        className="pretty_container six columns",
                                                        children=[dcc.Graph(
                                                                            id="loss-graph-hist",
                                                                            figure = go.Figure([])
                                                                            )
                                                                    ],
                                                        style={'display': 'none'}
                                                        ),
                                                html.Div(
                                                        id="task-metric-graph-con-hist",
                                                        className="pretty_container six columns",
                                                        children=[dcc.Graph(
                                                                            id="task-metric-graph-hist",
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
                                                        id="class-metric-graph-con-hist",
                                                        className="pretty_container six columns",
                                                        children=[dcc.Graph(
                                                                            id="class-metric-graph-hist",
                                                                            figure = go.Figure([])
                                                                            )
                                                                            ],                                                      
                                                        style={'display': 'none'}
                                                        ),
                                                html.Div(
                                                        id="conf-matrix-con-hist",
                                                        className="pretty_container six columns",
                                                        children=[
                                                                    dcc.Dropdown(
                                                                                id='conf-dropdown-hist',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                ),
                                                                    dcc.Graph(
                                                                                id="conf-matrix-hist",
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
                                                        id="text-highlight-con-hist",
                                                        className="pretty_container six columns",
                                                        children=[ 
                                                                    dcc.Dropdown(
                                                                                id='sample-id-dropdown1-hist',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                ),
                                                                    html.Img(   
                                                                            id="text-highlight-hist",
                                                                            src=""
                                                                            ),
                                                                    ],
                                                        ),
                                                html.Div(
                                                        id="tree-graph-con",
                                                        className="pretty_container six columns",
                                                        children=[
                                                                    dcc.Dropdown(
                                                                                id='sample-id-dropdown2-hist',
                                                                                options=[],
                                                                                value=None,
                                                                                className="dcc_control",
                                                                                ),
                                                                    dcc.Graph(
                                                                            id="tree-graph-hist",
                                                                            figure = go.Figure([])
                                                                            )
                                                                ],                                                         
                                                        ),
                                                ]
                                        )
                                ]
                            )
            
        exp_view = html.Div(   
                            id="exp-view",
                            #className="row flex-display",
                            children=[
                                        info,
                                        visuals
                                        ],
                            style={'display': 'none'}
                            #style={"display": "flex", "flex-direction": "column"},
                            )



        rank_view = html.Div(   
                            id="rank-view",
                            children=[
                                        dcc.Graph(
                                                    id="rank-graph",
                                                    figure = go.Figure([])
                                                    ),                                        
                                    ],
                            style={'display': 'none'}
                            )


        self.layout = html.Div(
                                children=[
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
                                                    children=[
                                                            dcc.Dropdown(
                                                                        id='task-filter',
                                                                        options=[],
                                                                        value=None,
                                                                        className="dcc_control",
                                                                        ),
                                                            dcc.Dropdown(
                                                                        id='split-filter',
                                                                        options=[
                                                                                {"label":"val", "value":"val"},
                                                                                {"label":"test", "value":"test"}
                                                                                ],
                                                                        value="val",
                                                                        className="dcc_control",
                                                                        ),                       
                                                            ],
                                                    style={'display': 'none'}
                                                    ),
                                            rank_view,
                                            dcc.Dropdown(
                                                        id='done-exp-dropdown',
                                                        options=[],
                                                        value=None,
                                                        className="dcc_control",
                                                        placeholder="Select a Experiment ID",
                                                        #persistence=True,
                                                        #clearable=False,
                                                        ),
                                            exp_view,
                                            html.Div(id='exp-data', children=dict(), style={'display': 'none'}),

                                          ]
                                
                                )


        app.callback(Output('project-dropdown', 'options'),
                    Input('interval-component', 'n_intervals'))(self.update_project_dropdown)
        

        app.callback(Output('rank-filters', 'style'),
                    Input('project-dropdown', 'value'))(self.show_rank_filters)
        

        app.callback(Output('task-filter', 'options'),
                    Input('project-dropdown', 'value'))(self.update_task_dropdown)


        app.callback(
                    Output('rank-graph', 'figure'),
                    Output('rank-view', 'style'),
                    [
                    Input('rank-graph', 'clickData'),
                    Input('project-dropdown','value'),
                    Input('task-filter', 'value'),
                    Input('split-filter', 'value')]
                    )(self.update_rank_graph)


        app.callback(Output('done-exp-dropdown', 'options'),
                    Input('interval-component', 'n_intervals'))(self.update_done_exp_dropdown)


        app.callback(Output('done-exp-dropdown', 'value'),
                    Input('rank-graph', 'clickData'),
                    State('done-exp-dropdown', 'value'))(self.update_exp_dropdown_value)


        app.callback(
                    Output('exp-data', 'children'),
                    Output('exp-view', 'style'),
                    Input('done-exp-dropdown','value'))(self.update_exp_data)

    

        app.callback(Output('exp-config-hist', 'children'),
                    [Input("exp-config-dropdown-hist", "value"),
                    Input('exp-data', 'children')])(self.get_config)


        app.callback(Output('data-table-hist', 'figure'),
                    [Input('exp-data', 'children')])(self.get_data_table)


        app.callback(
                    Output('loss-graph-hist', 'figure'),
                    Output('loss-graph-con-hist', 'style'),
                    [Input('exp-data', 'children')],
                    [State('loss-graph-hist', 'figure')])(self.update_loss_graph)


        app.callback(
                    Output('task-metric-graph-hist', 'figure'),
                    Output('task-metric-graph-con-hist', 'style'),
                    [Input('exp-data', 'children')],
                    [State('task-metric-graph-hist', 'figure')])(self.update_task_metric_graph)


        app.callback(
                    Output('class-metric-graph-hist', 'figure'),
                    Output('class-metric-graph-con-hist', 'style'),
                    [Input('exp-data', 'children')],
                    [State('class-metric-graph-hist', 'figure')])(self.update_class_metric_graph)


        app.callback(Output('conf-dropdown-hist', 'options'),
                    Output('conf-dropdown-hist', 'value'),
                    Input('done-exp-dropdown','value')
                    )(self.update_conf_dropdown)


        app.callback(
                    Output('conf-matrix-hist', 'figure'),
                    Output('conf-matrix-con-hist', 'style'),
                    [
                    Input('conf-dropdown', 'value'),
                    Input('exp-data', 'children')
                    ])(self.update_conf_matrix)

        
        app.callback(Output('sample-id-dropdown1-hist', 'options'),
                    Output('sample-id-dropdown1-hist', 'value'),
                    Input('done-exp-dropdown','value')
                    )(self.update_sample_id_dropdown)


        app.callback(
                    Output('text-highlight-hist', 'src'),
                    Output('text-highlight-con-hist', 'style'),
                    [
                    Input('sample-id-dropdown1-hist', 'value'),
                    Input('exp-data', 'children')
                    ])(self.update_highlight_text)


        app.callback(Output('sample-id-dropdown2-hist', 'options'),
                    Output('sample-id-dropdown2-hist', 'value'),
                    Input('done-exp-dropdown','value')
                    )(self.update_sample_id_dropdown)

        app.callback(
                    Output('tree-graph-hist', 'figure'),
                    Output('tree-graph-con-hist', 'style'),
                    [
                    Input('sample-id-dropdown2-hist', 'value'),
                    Input('exp-data', 'children')
                    ],
                    [State('tree-graph-hist', 'figure')])(self.update_tree_graph)


    def show_rank_filters(self, value):
        if value is not None:
            return {'display': 'block'}
        else:
            return {'display': 'none'}


    def update_exp_dropdown_value(self, clickData, state_value):
        if clickData:
            exp_id = dict(clickData)["points"][0]["customdata"]
            return exp_id
        else:
            return state_value
    

    def update_exp_data(self, experiment_id):
        filter_by = get_filter(experiment=experiment_id)
        last_epoch = self.db.get_last_epoch(filter_by)
        return self.get_exp_data(experiment_id, last_epoch)