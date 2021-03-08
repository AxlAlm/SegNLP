
#
import math
import numpy as np

#plotly
import plotly
import plotly.express as px
import plotly.graph_objs as go

#hotam
from hotam.dashboard.utils import fig_layout


def make_lineplot(data, title, max_y:int):

    if data.shape[0] == 0:
        return go.Figure(data=[])

    max_y = max_y
    max_x = data['epoch'].max()+1

    fig = go.Figure()
    
    data = data.sort_values("epoch")
    columns = [c for c in data.columns if c not in ["split", "epoch"]]
    for c in columns:
        
        metric_data = data.loc[:,[c,"split", "epoch"]]
        groups = metric_data.groupby("split")

        for split, df in groups:
            fig.add_trace({
                            'x': df['epoch'].to_numpy(),
                            'y': df[c].to_numpy(),
                            'name': f"{split}-{c}",
                            'mode': 'lines',
                            'type': 'scatter',
                            "connectgaps":True
                            })

    fig.update_layout(  
                        title=title,
                        yaxis=dict(range=[0,max_y]),
                        xaxis=dict(range=[0,max_x]),   
                        )

    return fig_layout(fig)