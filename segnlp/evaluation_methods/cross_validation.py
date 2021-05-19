#basics
import numpy as np
from copy import deepcopy

#pytorch
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

#segnlp
from segnlp.preprocessing.dataset_preprocessor import PreProcessedDataset
from segnlp.ptl import get_ptl_trainer_args
from segnlp.ptl import PTLBase

#pytorch lightning
from pytorch_lightning import Trainer 

def cross_validation(
                    model_args:dict,
                    ptl_trn_args:dict,
                    dataset:PreProcessedDataset,
                    save_choice:str
                    ):

    cv_scores = []
    model_fps = []
    for i in dataset.splits.keys():
        #dataset_cp = deepcopy(dataset)
        dataset.split_id = i

        trainer = Trainer(**ptl_trn_args)
        
        checkpoint_cb = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                new_filename = callback.filename + f"_fold={i}"
                setattr(callback, 'filename', new_filename)
                checkpoint_cb = callback

        ptl_model = PTLBase(**model_args)
        trainer.fit(    
                    model=ptl_model, 
                    train_dataloader=dataset.train_dataloader(), 
                    val_dataloaders=dataset.val_dataloader()
                    )

        if save_choice == "last":
            model_fp = checkpoint_cb.last_model_path
            checkpoint_dict = torch.load(model_fp)
            model_score = float(checkpoint_dict["callbacks"][ModelCheckpoint]["current_score"])
        else:
            model_fp = checkpoint_cb.best_model_path
            model_score = float(checkpoint_cb.best_model_score)

        cv_scores.append(model_score)
        model_fps.append(model_fp)
    
    mean_score = np.mean(cv_scores)

    return model_fps[0], mean_score
        