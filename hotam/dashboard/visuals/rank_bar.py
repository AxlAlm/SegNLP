
#basic
import numpy as np
import pandas as pd

#plotly
import plotly
import plotly.express as px
import plotly.graph_objs as go

#hotam
from hotam.dashboard.utils import fig_layout


def rank_bar(   
                title:str,
                top_scores:pd.DataFrame, 
                display_metrics:list, 
                display_splits:list, 
                experiment2config:dict, 
                rank_task:str, 
                top_n:int, 
                clickdata:dict):

    split2color = {
                    "test":'lightsalmon',
                    "val": "medal"
                    }
    
    bars = []
    for dm in display_metrics:
        for ds in display_splits:

            
            dmds = f"{ds}-{rank_task}{dm}"
            x = [i for i in range(top_n)]
            y = list(top_scores[dmds].to_numpy())
            text = [experiment2config[exp_id]["model"] for exp_id in top_scores["experiment_id"].to_numpy()]

            cd = list(top_scores["experiment_id"].to_numpy())
            b = go.Bar(
                        name=dmds, 
                        x=x, 
                        y=y,
                        text=text,
                        customdata=cd,
                        textposition='inside',
                        hovertemplate=
                        "<b>%{customdata}</b><br><br>" +
                        "Model: %{text}<br>" +
                        dm+": %{y}<br>" +
                        "<extra></extra>",                        
                        )
            bars.append(b)

    fig = go.Figure(data=bars)

    if clickdata:
        x1 = clickdata["points"][0]["x"] - 0.5
        x0 = x1+1
        fig.add_shape(
                    type="rect",
                    x0=x0,
                    y0=0,
                    x1=x1,
                    y1=1,
                    line=dict(
                                color="RoyalBlue",
                                )
                    )


    fig.update_layout(
                        barmode='group', 
                        title=title,
                        yaxis=dict(range=[0,1]),
                        clickmode='event'
                        )
    # fig.update_layout(
    #                     yaxis=dict(range=[0,1]),
    #                     #xaxis=dict(range=[0,max_x]),   
    #                     )
    # fig.update_layout(clickmode='event')

    return fig