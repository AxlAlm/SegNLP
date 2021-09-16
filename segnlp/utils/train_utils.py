
#basics
import os

#pytroch
import torch


class EarlyStopping:

    def __init__(self, patience:int):
        self._patience = patience
        self._prev_score = 0
        self._n_epochs_wo_improvement = 0


    def ___call__(self, score):

        if self._patience is None:
            return False

        if score < self._prev_score:
            self._n_epochs_wo_improvement += 1
        else:
            self._n_epochs_wo_improvement = 0

        if self._n_epochs_wo_improvement > self._patience:
            return True
   


class SaveBest:

    def __init__(self, path_to_model:str):
        self._path_to_model = os.path.join(path_to_model)
        self._prev_score = 0


    def ___call__(self, model: torch.nn.Module, score:float):

        if score > self._prev_score:
            torch.save(model, self._path_to_model)
            self._prev_score = score