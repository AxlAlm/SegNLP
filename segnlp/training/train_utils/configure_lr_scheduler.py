

# pytorch
from typing import Union
import torch
from torch import optim
from torch.optim import lr_scheduler


def configure_lr_scheduler(self, opt : optim.Optimizer, config : dict) -> Union[None, lr_scheduler._LRScheduler]:
    
    # if we are not using any return None
    if config is None:
        return None

    lrs_name = config.pop("name")

    # find the learning scheduler
    lr_s_class = getattr(torch.optim.lr_scheduler, lrs_name)

    # init the scheduler
    lr_s = lr_s_class(opt, **config)

    return lr_s
 
