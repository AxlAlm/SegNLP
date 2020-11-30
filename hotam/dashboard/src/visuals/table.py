
import numpy as np

#plotly
import plotly
import plotly.express as px
import plotly.graph_objs as go


from src.utils import fig_layout


def make_table(data, title):
    fig = go.Figure(
                    data=[go.Table( 
                                    header=dict(values=["type"] + list(data.columns),
                                                line_color='darkslategray',
                                                fill_color='lightskyblue',
                                                align='left'),
                                    cells=dict(values=[data.index] + [c_data for _, c_data in data.iteritems()],
                                                line_color='darkslategray',
                                                fill_color='lightcyan',
                                                align='left'),
                                    columnwidth=[300, 100, 100, 100]
                                    )
                        ]
                    )
    fig.update_layout(
                        title=title,
                        )

    return fig