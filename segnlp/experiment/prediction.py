
# basics
from typing import List
import os
from glob import glob
from copy import deepcopy

# pytorch
import torch

# segnlp
from segnlp.data import Sample
from segnlp.data import Batch
from segnlp.seg_model import SegModel
from segnlp import utils



class Predict:


    def prediction_mode(self, model:SegModel) -> None:
      

        # load hyperparamaters
        hp_config = utils.load_json(os.path.join(self._path_to_hps, model.name(), "0.json"))



        hyperparamaters = hp_config["hyperparamaters"]
        path_to_models = hp_config["path_to_models"]
        random_seed = hp_config["random_seeds_done"][0]
        model_path = glob(os.path.join(path_to_models, f"{random_seed}*.ckpt"))[0]

        #setup model
        self._pred_model = model(
                            hyperparamaters  = hyperparamaters,
                            task_dims = self.dataset.task_dims,
                            seg_task = self.dataset.seg_task
                            )

        #load model weights
        self._pred_model.load_state_dict(torch.load(model_path))



    def predict(self, list_samples: List[Sample], batch_size:int = 10) -> List[Sample]:
        pred_samples = []
        for i in range(0,len(list_samples), batch_size):
            batch = Batch(list_samples[i: i+batch_size])
            self._pred_model(batch)
            pred_samples.extend(batch._pred_samples)

        return pred_samples
