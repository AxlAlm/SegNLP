


import plotly.graph_objects as go
import numpy as np
import pandas as pd
import re

def make_rel_error_dist_plot(data, data_stats, splits, filter_columns):

    

    stats_df = pd.DataFrame(data_stats)
    stats_df = stats_df[stats_df["task"] == "relation"]

    scores_df  = pd.DataFrame(data)

    splits_stats = stats_df.groupby("split")
    splits_scores = scores_df.groupby("split")

    sorted_columns = sorted(filter_columns, key=lambda x:int(re.sub(r"relation-|-f1", "", x)))
    
    last_epoch = scores_df["epoch"].max()

    fig = go.Figure()    

    for s in splits:

        sdf = splits_stats.get_group(s)
        relations = sdf["type"].to_numpy()
        counts = sdf["value"].to_numpy()
        _sum = sum(counts)
        dist = [c/_sum for c in counts]
        fig.add_trace(go.Scatter(
                                x=relations, 
                                y=dist,
                                mode='lines',
                                name=s,
                                hovertext=counts,
                                ))

        
        sdf_scores = splits_scores.get_group(s)
        cond = sdf_scores["epoch"]==last_epoch
        last_epoch_scores = sdf_scores.loc[cond]
        scores = last_epoch_scores.loc[:,sorted_columns].T
        filtered_last_epoch_scores = scores[last_epoch].to_numpy()

        fig.add_trace(go.Scatter(
                        x=relations, 
                        y=filtered_last_epoch_scores,
                        mode='lines',
                        name=f"{s}-f1",
                        hovertext=counts,
                        ))


    fig.update_layout(  
                        title="F1 across relation distance",
                        yaxis=dict(range=[0,1]),
                        xaxis=dict(range=[-11,10]),   
                        )
    # fig.add_annotation(	
    #                     x=_min,
    #                     xanchor="left",
    #                     yanchor="top",
    #                     y=smax,
    #                     text=f"TOTAL:\n {total_string}",
    #                     showarrow=False,
    #                     )

    #fig.update_layout(title="Distribution of relations as measured in number of Argument Components back or forward")
    return fig