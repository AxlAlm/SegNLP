
# basics
from copy import deepcopy

# pytorch
import torch
from torch import optim
import torch.nn as nn


def configure_optimizer(model: nn.Module, config:dict) -> optim.Optimizer:

    config = deepcopy(config)

    # get the name
    opt_name = config.pop("name")

    # get a optimizer from torch.optim
    opt_class = getattr(torch.optim, opt_name)

    # init the optimizer
    opt = opt_class(model.parameters(), **config)

    return opt