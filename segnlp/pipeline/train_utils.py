

# basics
from typing import Tuple, Union

# pytroch
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler


class TrainUtils:


    def _configure_optimizers(self, model: nn.Module, hyperparamaters:dict) -> optim.Optimizer:

        #find the optimizer hyperparamaters or str
        opt_hp = hyperparamaters["general"]["optimizer"]

        # find the optimizer and fix kwargs 
        if isinstance(opt_hp, dict):
            opt_class = getattr(torch.optim, opt_hp.pop("name"))
        else:
            opt_class = getattr(torch.optim, hyperparamaters["general"]["optimizer"])
            opt_hp = {}

        # init the optimizer
        opt = opt_class(model.parameters(), **opt_hp)

        return opt


    def _configure_learning_scheduler(self, opt : optim.Optimizer, hyperparamaters : dict) -> Union[None, lr_scheduler._LRScheduler]:
        
        # get the hyperparamaters for learnign scheduler or the str
        lr_s_hps =  hyperparamaters["general"].get("lr_scheduler", None)

        # if we are not using any return None
        if lr_s_hps is None:
            return None

        # find the learning scheduler and fix kwargs 
        if isinstance(lr_s_hps, dict):
            lr_s_class = getattr(torch.optim.lr_scheduler, lr_s_hps.pop("name"))
        else:
            lr_s_class = getattr(torch.optim.lr_scheduler, lr_s_hps)
            lr_s_hps = {}

        # init the scheduler
        lr_s = lr_s_class(opt, **lr_s_hps)

        return lr_s
 
