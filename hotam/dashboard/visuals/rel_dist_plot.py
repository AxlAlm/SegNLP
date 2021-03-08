import plotly.graph_objects as go
import numpy as np

def make_relation_dist_plot(df):

    splits = df.groupby("split")

    _min = 0
    _max = 1
    fig = go.Figure()
    total_string = ""
    for s, sdf in splits:
        relations = sdf["label"].to_numpy().astype(float)

        counts = sdf["count"].to_numpy()
        _sum = np.sum(counts)
        dist = [c/_sum for c in counts]

        smax = max(dist)
        if smax > _max:
            _max = smax

        smin = min(relations)
        if smin < _min:
            _min = smin

        total_string += f"{s}: {_sum}\n"
        fig.add_trace(go.Scatter(
                                x=relations, 
                                y=dist,
                                mode='markers+lines',
                                name=s,
                                hovertext=counts,
                                ))

    fig.add_annotation(	
                        x=_min,
                        xanchor="left",
                        yanchor="top",
                        y=smax,
                        text=f"TOTAL:\n {total_string}",
                        showarrow=False,
                        )

    fig.update_layout(title="Distribution of relations as measured in number of Argument Components back or forward")
    return fig