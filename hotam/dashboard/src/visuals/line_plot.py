

import math
import numpy as np

#plotly
import plotly
import plotly.express as px
import plotly.graph_objs as go


from src.utils import fig_layout



def make_lineplot(data, columns, title):

    if data.shape[0] == 0:
        return go.Figure(data=[])

    grouped = data.groupby("split")
    max_y = 0
    max_x = max(grouped.get_group("val")['epoch'])+1

    fig = go.Figure()

    for c in columns:

        if c not in grouped.get_group("train").columns:
            continue

        tc = grouped.get_group("train")[c].max()
        vc = grouped.get_group("val")[c].max()

        if tc > max_y:
                max_y = tc
        
        if vc > max_y:
            max_y = vc

        fig.add_trace({
            'x': grouped.get_group("train")['epoch'].to_numpy(),
            'y': grouped.get_group("train")[c].to_numpy(),
            'name': f"train-{c}",
            'mode': 'lines',
            'type': 'scatter',
            "connectgaps":True
        })

        fig.add_trace({
            'x': grouped.get_group("val")['epoch'].to_numpy(),
            'y': grouped.get_group("val")[c].to_numpy(),
            'name': f"val-{c}",
            'mode': 'lines',
            'type': 'scatter',
            "connectgaps":True

        })

    
    fig.update_layout(  
                        title=title,
                        yaxis=dict(range=[0,np.ceil(max_y)]),
                        xaxis=dict(range=[0,max_x]),   
                        )

 
    return fig_layout(fig)