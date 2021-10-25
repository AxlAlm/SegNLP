

#basics
import os
from copy import deepcopy

#pytroch
import torch


class SaveBest:

    def __init__(self, model_file:str) -> None:
        self._model_file = model_file
        self._top_score = 0


    def __call__(self, model: torch.nn.Module, score:float) -> None:

        if score > self._top_score:
            state_dict = deepcopy(model.state_dict())
            torch.save(state_dict, self._model_file)
            self._top_score = score