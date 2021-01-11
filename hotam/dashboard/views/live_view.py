
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
from .base import ViewBase


th_path  = "/tmp/text_highlight.png"

class LiveView(ViewBase):

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
                                                        #persistence=True,
                                                        #clearable=False,
                                                        ),
                                            live_view,
                                            html.Div(id='data-cache', children=dict(), style={'display': 'none'}),

                                          ]
                                
                                )

        
        app.callback(Output('exp-dropdown', 'options'),
                    Input('interval-component', 'n_intervals'))(self.update_live_exp_dropdown)


        app.callback(
                    Output('data-cache', 'children'),
                    Output('live-view', 'style'),
                    [Input('interval-component', 'n_intervals'),
                    Input('exp-dropdown','value')],
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

        return self.get_exp_data(experiment_id, last_epoch)
