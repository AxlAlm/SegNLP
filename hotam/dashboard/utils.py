

def get_filter(dataset=None, project=None, model=None, experiment=None):
    filter_by = {}

    if isinstance(dataset,str) and dataset:
        filter_by["dataset"] = dataset

    if isinstance(project, str) and project:
        filter_by["project"] = project

    if isinstance(model, str) and model:
        filter_by["model"] = model

    if isinstance(experiment, str) and experiment:
        filter_by["experiment_id"] = experiment
    
    return filter_by


def fig_layout(fig):

    fig.update_layout(dict(
                                autosize=True,
                                #automargin=True,
                                margin=dict(l=150, r=30, b=20, t=40),
                                hovermode="closest",
                                plot_bgcolor="#F9F9F9",
                                paper_bgcolor="#F9F9F9",
                                #legend=dict(font=dict(size=10), orientation="h"),
                                #title=title,
                            )
                    )
    fig.update_yaxes(automargin=True)

    return fig

def get_visible_info(fig):
    visible_info = {}
    fig.for_each_trace(lambda trace: visible_info.update({trace.name:trace.visible})) 
    return visible_info


def update_visible_info(fig, visible_info):
    fig.for_each_trace(lambda trace: trace.update(visible=visible_info[trace.name])) 
