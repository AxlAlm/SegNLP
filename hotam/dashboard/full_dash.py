
#dash
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input

# import app and mongo db
from src.servers import app, db


from src.exp_view import exp_sides
from src.task_view import overview_sides
#from src.hypertune import hp_sides


# Initialise the 
app.layout = html.Div([
                        dcc.Tabs(
                                id='tabs', 
                                value='tab-2', 
                                children=[
                                            dcc.Tab(label='Overview', value='tab-1'),
                                            dcc.Tab(label='Experiment', value='tab-2'),
                                            ]
                                ),
                        html.Div(id='tab-content')
                    ])



@app.callback(Output('tab-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-2':
        return html.Div(
                        children=[
                                    exp_sides,
                                    dcc.Interval(
                                                    id='interval-component',
                                                    interval=1*1000, # in milliseconds
                                                    n_intervals=0,
                                                    max_intervals=-1,
                                                )  
                                ]
                    )
    elif tab == 'tab-1':
        return html.Div(
                        children=[  
                                    overview_sides,
                                    dcc.Interval(
                                                    id='interval-component2',
                                                    interval=1*5000, # in milliseconds
                                                    n_intervals=0,
                                                    max_intervals=-1,
                                                )
                                ]
                        )
    # elif tab == 'tab-3':
    #     return html.Div(
    #                     children=[
    #                                 hp_sides,
    #                             ]
    #                     )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


import dash
from data.mongo_data import MongoData

#init the mongo db
db = MongoData()

# Initialise the app
app = dash.Dash(__name__)
