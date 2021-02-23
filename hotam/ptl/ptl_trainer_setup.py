#basic
import os
import copy
from glob import glob
import re
from typing import Tuple, List, Dict

#pytorch lightning 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer 


def get_ptl_trainer(self, experiment_id, trainer_args, hyperparamaters, model_dump_path:str=None, verbose=0):
    

    if trainer_args.get("checkpoint_callback",False) == True:
        trainer_args["checkpoint_callback"] = ModelCheckpoint(
                                                                filepath=model_dump_path,
                                                                save_top_k=trainer_args["save_top_k"],
                                                                verbose=False if verbose==0 else True,
                                                                monitor=hyperparamaters["monitor_metric"],
                                                                mode='min' if "loss" in hyperparamaters["monitor_metric"] else "max",
                                                                prefix=experiment_id,
                                                                )

    if trainer_args.get("early_stop_callback",False) == True:
        trainer_args["early_stop_callback"] = EarlyStopping(
                                                            monitor=hyperparamaters["monitor_metric"], 
                                                            patience=hyperparamaters["patience"],
                                                            mode='min' if "loss" in hyperparamaters["monitor_metric"] else "max",
                                                            verbose=False if verbose==0 else True,

                                                            )


    #overwrite the Pytroch Lightning Training arguments that are writen in Hyperparamaters 
    if "max_epochs" in trainer_args or "max_epochs" in hyperparamaters:
        trainer_args["max_epochs"] = hyperparamaters["max_epochs"]
    
    if "gradient_clip_val" in trainer_args or "gradient_clip_val" in hyperparamaters:
        trainer_args["gradient_clip_val"] = hyperparamaters["gradient_clip_val"]
                    

    trainer_args["default_root_dir"] = model_dump_path
    trainer = Trainer(**trainer_args)
    return trainer


