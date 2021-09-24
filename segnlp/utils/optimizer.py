

# basics
from typing import Tuple, Union

# pytroch
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler


def configure_optimizers(model: nn.Module, hyperparamaters:dict) -> Tuple[optim.Optimizer, Union[None, 
                                                                                lr_scheduler._LRScheduler]]:

    opt_class = getattr(torch.optim, hyperparamaters["general"]["optimizer"])
    opt = opt_class(model.parameters(), lr = hyperparamaters["general"]["lr"])

    lr_s = None
    if "lr_scheduler" in hyperparamaters["general"]:
        lr_s = getattr(torch.optim.lr_scheduler , hyperparamaters["general"]["lr_scheduler"])
        lr_s = lr_s(opt, **hyperparamaters["general"].get("lr_scheduler_kwargs", {}))

    return opt, lr_s

