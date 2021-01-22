

import plotly.graph_objects as go


def label_dist_plot(df):
    types = df.groupby("type")
    
    bars = []
    for t, tdf in types:
        x = tdf["split"].to_numpy()
        y = tdf["value"].to_numpy()
        bars.append(go.Bar(
                            name=t,
                            x=x, 
                            y=y,
                            #text=[t,t,t],
                            #textposition='inside',
                            ))

    fig = go.Figure(data=bars)
    fig.update_layout(barmode='group')
    fig.update_layout(title=f"Label Distribution")
    return fig
