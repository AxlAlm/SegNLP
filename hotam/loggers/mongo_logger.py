


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

logger = get_logger("MONGO-LOGGER")


class MongoLogger(LightningLoggerBase):


    def __init__(self, db):
        super().__init__()
        self.db = db
        self.current_epoch = 0
        self.output_stack = {}
    
    def __repr__(self):
        return self.name()

    def name(self):
        return "MongoLogger"


    def version(self):
        return 0.1
    

    def log_experiment( self, experiment_config:dict):
        self.db.experiments.insert_one(copy_and_vet_dict(experiment_config))


    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics["experiment_id"] = self.experiment_id
        split = [k for k in metrics.keys() if "epoch" not in k and "id" not in k][-1].split("-",1)[0]
        metrics["split"] = split
        self.db.scores.insert_one(copy_and_vet_dict(metrics, filter_key=split+"-"))

        
        #also logs the stack of outputs for the epoch
        if split in ["val", "test"]:
            self.db.outputs.insert_one({ 
                                        "experiment_id": self.experiment_id,
                                        "epoch": metrics["epoch"],
                                        "split": split,
                                        "data": self.output_stack
                                        })
            self.output_stack = {}


    def set_exp_id(self, experiment_id:str):
        self.experiment_id = experiment_id
    

    def log_outputs(self, outputs):
        self.output_stack.update(outputs)

    
    def update_config(self, experiment_id, key, value):
        newvalues = { "$set": { key: value} }
        self.db.experiments.update_one({"experiment_id":experiment_id}, newvalues)
    
    # def delete(self, experiment_id):
    #     filter_by = {"experiment_id":experiment_id}
    #     self.db.experiments.delete_many(filter_by)

    def experiment(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        pass
