


#basics 
import os
from glob import glob
import re
from pathlib import Path
import numpy as np
import json
from copy import deepcopy

#mongodb
import pymongo 

#pytorch lightning
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase

#hotam
from hotam import get_logger
from hotam.utils import copy_and_vet_dict

logger = get_logger("LocalLogger")


class LocalLogger(LightningLoggerBase):

    """
    structure

    root_dir/
        experiments/
            <exp_id>_<project>_<dataset>_<model>_<timestamp>/
                config.json
                scores/
                    val/
                        epoch=1.json
                        epoch=n.json
                        ..
                    train/
                        epoch=n.json
                outputs/
                    epoch=1.json
                    epoch=2.json
                    ..


    """

    def __init__(self, root_dir="~/.hotam/"):
        super().__init__()
        self.root_dir = root_dir
        self.experiment_dir = os.path.join(self.root_dir, "experiments")
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def __repr__(self):
        return self.name()

    def name(self):
        return "LocalLogger"


    def version(self):
        return 0.1
    

    def log_experiment( self, experiment_config:dict):
        folder_name = "_".join([
                                experiment_config["experiment_id"],
                                experiment_config["project"],
                                experiment_config["dataset"],
                                experiment_config["model"],
                                experiment_config["start_time"]
                                ])

        self.exp_folder = os.path.join(self.experiment_dir, folder_name, exist_ok=True)
        self.score_folder  = os.path.join(self.exp_folder, "scores", exist_ok=True)
        self.output_folder  = os.path.join(self.exp_folder, "outputs", exist_ok=True)

        self.val_scores  = os.path.join(self.score_folder, "val", exist_ok=True)
        self.train_scores  = os.path.join(self.score_folder, "train", exist_ok=True)

        os.makedirs(self.exp_folder, exist_ok=True)
       # os.makedirs(self.score_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.val_scores, exist_ok=True)
        os.makedirs(self.train_scores, exist_ok=True)


        config_file_path = os.path.join(self.exp_folder, "config.json")
        with open(config_file_path, "w") as f:
            json.dump(experiment_config, f, indent=4)


    @rank_zero_only
    def log_metrics(self, metrics:dict, epoch:int, split:str):
        metrics["experiment_id"] = self.experiment_id
        metrics["split"] = split,
        metrics["epoch"] = epoch

        score_file = os.path.join(self.val_scores, f"epoch={epoch}.json")
        with open(score_file, "w") as f:
            json.dump(metrics, f, indent=4)


    def set_exp_id(self, experiment_id:str):
        self.experiment_id = experiment_id
    

    def log_outputs(self, outputs):
        self.output_stack.update(outputs)

    
    def update_config(self, experiment_id, key, value):

        config_file_path = os.path.join(self.exp_folder, "config.json")

        with open(config_file_path, "r") as f:
            config = json.load(f)

        config[key] = value

        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=4)
    

    def experiment(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        pass
