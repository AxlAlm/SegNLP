

import plotly.graph_objects as go


def label_dist_plot(df):
    label_counts = df.groupby("label")
    
    bars = []
    for l, ldf in label_counts:
        splits = ldf.groupby("label")
        for s, sdf in splits:
            x = sdf["split"].to_numpy()
            y = sdf["count"].to_numpy()
            bars.append(go.Bar(
                                name=l,
                                x=x, 
                                y=y,
                                legendgroup=s,
                                #text=[t,t,t],
                                #textposition='inside',
                                ))

    fig = go.Figure(data=bars)
    fig.update_layout(barmode='group')
    fig.update_layout(title=f"Label Distribution")
    return fig
