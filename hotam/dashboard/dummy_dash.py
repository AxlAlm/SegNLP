
#dash
import dash
import dash_html_components as html
import dash_core_components as dcc

#hotam
from hotam.dashboard.views.exp_view import ExpView


class DummyDash:

    def __init__(self, db):
        self.app = dash.Dash("Dummy Dash Board")    
        self.app.layout = html.Div([  
                                    ExpView(self.app, db).layout,
                                    dcc.Interval(
                                                    id='interval-component',
                                                    interval=1*5000, # in milliseconds
                                                    n_intervals=0,
                                                    max_intervals=-1,
                                                )
                                     ])

    def run_server(self, *args, **kwargs):
        self.app.run_server(*args, **kwargs)


