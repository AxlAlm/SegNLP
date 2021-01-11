
# #dash
# import dash
# import dash_html_components as html
# import dash_core_components as dcc
# from dash.dependencies import Output, Input

# # import app and mongo db
# from src.servers import app, db


# from src.exp_view import exp_sides
# from src.task_view import overview_sides
# #from src.hypertune import hp_sides


# # Initialise the 
# app.layout = html.Div([
#                         dcc.Tabs(
#                                 id='tabs', 
#                                 value='tab-2', 
#                                 children=[
#                                             dcc.Tab(label='Overview', value='tab-1'),
#                                             dcc.Tab(label='Experiment', value='tab-2'),
#                                             ]
#                                 ),
#                         html.Div(id='tab-content')
#                     ])



# @app.callback(Output('tab-content', 'children'),
#               [Input('tabs', 'value')])
# def render_content(tab):
#     if tab == 'tab-2':
#         return html.Div(
#                         children=[
#                                     exp_sides,
#                                     dcc.Interval(
#                                                     id='interval-component',
#                                                     interval=1*1000, # in milliseconds
#                                                     n_intervals=0,
#                                                     max_intervals=-1,
#                                                 )  
#                                 ]
#                     )
#     elif tab == 'tab-1':
#         return html.Div(
#                         children=[  
#                                     overview_sides,
#                                     dcc.Interval(
#                                                     id='interval-component2',
#                                                     interval=1*5000, # in milliseconds
#                                                     n_intervals=0,
#                                                     max_intervals=-1,
#                                                 )
#                                 ]
#                         )
#     # elif tab == 'tab-3':
#     #     return html.Div(
#     #                     children=[
#     #                                 hp_sides,
#     #                             ]
#     #                     )


# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)


# import dash
# from data.mongo_data import MongoData

# #init the mongo db
# db = MongoData()

# # Initialise the app
# app = dash.Dash(__name__)

#basic
from multiprocessing import Process


#dash
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import dash_daq as daq

#hotam
from .views.live_view import LiveView
from .views.hist_view import HistView


class FullDash:

    def __init__(self, db):
        self.app = dash.Dash("Dummy Dash Board")  
        self.live_view = LiveView(self.app, db).layout  
        self.hist_view = HistView(self.app, db).layout
        self.app.layout = html.Div([
                                    dcc.Tabs(
                                            id='tabs', 
                                            value='tab-1', 
                                            children=[
                                                        dcc.Tab(label='hist-view', value='tab-1'),
                                                        dcc.Tab(label='live-view', value='tab-2'),
                                                        ]
                                            ),
                                    html.Div(id='tab-content')
                                ])

        self.app.callback(Output('tab-content', 'children'),
                            [Input('tabs', 'value')])(self.render_content)


    def render_content(self, tab):
        if tab == 'tab-1':
            return html.Div(
                            children=[
                                        self.hist_view,
                                        dcc.Interval(
                                                        id='interval-component',
                                                        interval=1*10000, # in milliseconds
                                                        n_intervals=0,
                                                        max_intervals=-1,
                                                    )  
                                    ]
                        )
        elif tab == 'tab-2':
            return html.Div(
                            children=[  
                                        self.live_view,
                                        dcc.Interval(
                                                        id='interval-component2',
                                                        interval=1*1000, # in milliseconds
                                                        n_intervals=0,
                                                        max_intervals=-1,
                                                    )
                                    ]
                            )



    def run_server(self, *args, **kwargs):
        self.app.run_server(*args, **kwargs)
        # dashboard = Process(
        #                         target=self.app.run_server, 
        #                         args=args,
        #                         kwargs=kwargs,
        #                     )
        # dashboard.start()

