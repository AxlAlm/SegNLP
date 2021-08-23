#basic
import os
import copy
from glob import glob
import re
from typing import Tuple, List, Dict

#pytorch lightning 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ProgressBar


def get_ptl_trainer_args(
                        ptl_trn_args:dict, 
                        hyperparamaters:dict, 
                        save_choice:str,
                        exp_model_path:str,
                        
                        ):

    default_ptl_trn_args = dict(
                                logger = None,
                                progress_bar_refresh_rate = 1,
                                weights_summary = None
                                )
    
    ptl_trn_args = {**default_ptl_trn_args, **ptl_trn_args}

    # for adding costum callbacks. We add ProgressBar only to set the process_position
    ptl_trn_args["callbacks"] = [
                                ProgressBar(
                                            refresh_rate=ptl_trn_args["progress_bar_refresh_rate"], 
                                            process_position=3
                                            ),
                                ]

    if save_choice:
        
        if len([c for c in ptl_trn_args["callbacks"] if isinstance(c, ModelCheckpoint)]) == 0:

            save_top_k = 1

            if save_choice == "all":
                save_top_k = -1

            monitor_metric = hyperparamaters["general"]["monitor_metric"]

            os.makedirs(exp_model_path, exist_ok=True)
            mc  = ModelCheckpoint(
                                    dirpath=exp_model_path,
                                    filename='{epoch}-{val_loss:.2f}-{val_f1:.2f}',
                                    save_last=True if save_choice == "last" else False,
                                    save_top_k=save_top_k,
                                    monitor=monitor_metric,
                                    mode='min' if "loss" in monitor_metric else "max",
                                    #prefix=prefix,
                                    verbose=0,

                                    )

            ptl_trn_args["callbacks"].append(mc)


    if "patience" in hyperparamaters["general"]:

        if ptl_trn_args["callbacks"] == None:
            ptl_trn_args["callbacks"] = []

        if len([c for c in ptl_trn_args["callbacks"] if isinstance(c, EarlyStopping)]) == 0:
            ptl_trn_args["callbacks"].append(EarlyStopping(
                                                                monitor=hyperparamaters["general"]["monitor_metric"], 
                                                                patience=hyperparamaters["general"]["patience"],
                                                                mode='min' if "loss" in hyperparamaters["general"]["monitor_metric"] else "max",
                                                                verbose=0,
                                                                ))

    #overwrite the Pytroch Lightning Training arguments that are writen in Hyperparamaters 
    if "max_epochs" in ptl_trn_args or "max_epochs" in hyperparamaters["general"]:
        ptl_trn_args["max_epochs"] = hyperparamaters["general"]["max_epochs"]
    
    if ("gradient_clip_val" in ptl_trn_args and  "gradient_clip_val" == None) or  "gradient_clip_val" in hyperparamaters["general"]:
        ptl_trn_args["gradient_clip_val"] = hyperparamaters["general"]["gradient_clip_val"]
                    

    ptl_trn_args["default_root_dir"] = exp_model_path
    #trainer = Trainer(**ptl_trn_args)

    # if ptl_trn_args["callbacks"]:
    #     ptl_trn_args["callbacks"] = [str(c) for c in ptl_trn_args["callbacks"]]

    return ptl_trn_args


