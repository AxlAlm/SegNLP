

# pytroch
import torch


class Optim:


    def _configure_optimizers(self, hyperparamaters):

        opt_class = getattr(torch.optim, hyperparamaters["general"]["optimizer"])
        opt = opt_class(self.parameters(), lr = hyperparamaters["general"]["lr"])

        lr_s = None
        if "lr_scheduler" in hyperparamaters["general"]:
            lr_s = getattr(torch.optim.lr_scheduler , shyperparamaters["general"]["lr_scheduler"])
            lr_s = lr_s(opt, hyperparamaters["general"].get("lr_scheduler_kwargs", {}))

        return opt, lr_s

