

from src.servers import db


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


def dropdown_filter(name, dataset=None, project=None, model=None):
    filter_by = get_filter(dataset=dataset, project=project, model=model)
    return db.get_dropdown_options(name, filter_by=filter_by)


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