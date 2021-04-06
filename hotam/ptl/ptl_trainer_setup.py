#basic
import os
import copy
from glob import glob
import re
from typing import Tuple, List, Dict

#pytorch lightning 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer 

default_ptl_trn_args = dict(
                            logger=None, 
                            checkpoint_callback=None, 
                            callbacks=None, 
                            default_root_dir=None, 
                            gradient_clip_val=0, 
                            process_position=0, 
                            num_nodes=1, 
                            num_processes=1, 
                            gpus=None, 
                            auto_select_gpus=False, 
                            tpu_cores=None, 
                            log_gpu_memory=None, 
                            progress_bar_refresh_rate=1, 
                            overfit_batches=0.0, 
                            track_grad_norm=-1, 
                            check_val_every_n_epoch=1, 
                            fast_dev_run=False,
                            accumulate_grad_batches=1, 
                            max_epochs=1000, 
                            min_epochs=1, 
                            max_steps=None, 
                            min_steps=None, 
                            limit_train_batches=1.0, 
                            limit_val_batches=1.0, 
                            limit_test_batches=1.0, 
                            val_check_interval=1.0, 
                            flush_logs_every_n_steps=100, 
                            log_every_n_steps=50, 
                            accelerator=None, 
                            sync_batchnorm=False,
                            precision=32, 
                            weights_summary='top', 
                            weights_save_path=None, 
                            num_sanity_val_steps=2, 
                            truncated_bptt_steps=None, 
                            resume_from_checkpoint=None, 
                            profiler=None, 
                            benchmark=False, 
                            deterministic=False, 
                            reload_dataloaders_every_epoch=False, 
                            auto_lr_find=False, 
                            replace_sampler_ddp=True, 
                            terminate_on_nan=False, 
                            auto_scale_batch_size=False, 
                            prepare_data_per_node=True, 
                            plugins=None, 
                            amp_backend='native', 
                            amp_level='O2', 
                            distributed_backend=None, 
                            automatic_optimization=None, 
                            move_metrics_to_cpu=False, 
                            enable_pl_optimizer=None
                            )


def setup_ptl_trainer(
                        ptl_trn_args:dict, 
                        hyperparamaters:dict, 
                        save_choice:str,
                        model_dump_path:str,
                        prefix:str,
                        ):



    if save_choice:

        ptl_trn_args["callbacks"] = []

        save_top_k = 1
        # if save_choice == "best":
        #     save_top_k = 1

        if save_choice == "all":
            save_top_k = -1

        monitor_metric = hyperparamaters["monitor_metric"]

        os.makedirs(model_dump_path, exist_ok=True)
        mc  = ModelCheckpoint(
                                dirpath=model_dump_path,
                                save_last=True if save_choice == "last" else False,
                                save_top_k=save_top_k,
                                monitor=monitor_metric,
                                mode='min' if "loss" in monitor_metric else "max",
                                prefix=prefix,
                                verbose=0,

                                )

        ptl_trn_args["callbacks"].append(mc)

    # if "early_stop":
    #     if trainer_args["callbacks"] == None:
    #         trainer_args["callbacks"] = []

    #     trainer_args["callbacks"].append(EarlyStopping(
    #                                                         monitor=hyperparamaters["monitor_metric"], 
    #                                                         patience=hyperparamaters["patience"],
    #                                                         mode='min' if "loss" in hyperparamaters["monitor_metric"] else "max",
    #                                                         verbose=0,
    #                                                         ))


    #overwrite the Pytroch Lightning Training arguments that are writen in Hyperparamaters 
    if "max_epochs" in ptl_trn_args or "max_epochs" in hyperparamaters:
        ptl_trn_args["max_epochs"] = hyperparamaters["max_epochs"]
    
    if ("gradient_clip_val" in ptl_trn_args and  "gradient_clip_val" == None) or  "gradient_clip_val" in hyperparamaters:
        ptl_trn_args["gradient_clip_val"] = hyperparamaters["gradient_clip_val"]
                    

    ptl_trn_args["default_root_dir"] = model_dump_path
    trainer = Trainer(**ptl_trn_args)

    if ptl_trn_args["callbacks"]:
        ptl_trn_args["callbacks"] = [str(c) for c in ptl_trn_args["callbacks"]]

    return trainer


