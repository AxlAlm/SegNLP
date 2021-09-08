#basics
import numpy as np
from copy import deepcopy

#pytorch
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

#segnlp
from segnlp.utils import DataModule

#pytorch lightning
from pytorch_lightning import Trainer  as PTL_Trainer



class Evaluator:


    def __cross_validation(self,
                            model_args:dict,
                            ptl_trn_args:dict,
                            data_module:DataModule,
                            ):

        cv_scores = []
        model_fps = []
        for i in data_module.splits.keys():
            data_module.change_split_id(i)

            trainer = PTL_Trainer(**ptl_trn_args)
            
            checkpoint_cb = None
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    new_filename = callback.filename + f"_fold={i}"
                    setattr(callback, 'filename', new_filename)
                    checkpoint_cb = callback

            model = self.model(**model_args)
        
            trainer.fit(    
                        model = model, 
                        train_dataloaders = data_module.train_dataloader(), 
                        val_dataloaders = data_module.val_dataloader()
                        )

            model_fp = checkpoint_cb.best_model_path
            model_score = float(checkpoint_cb.best_model_score)

            cv_scores.append(model_score)
            model_fps.append(model_fp)
        
        mean_score = np.mean(cv_scores)

        return model_fps[0], mean_score
            

    def __default(
                self,
                model_args:dict,
                ptl_trn_args:dict,
                data_module:DataModule,
                ):

        trainer = PTL_Trainer(**ptl_trn_args)
        model = self.model(**model_args)

        trainer.fit(    
                        model = model, 
                        train_dataloaders = data_module.train_dataloader(), 
                        val_dataloaders = data_module.val_dataloader()
                        )

        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                model_fp = callback.best_model_path
                model_score = float(callback.best_model_score)
            
        return model_fp, model_score


    def _eval(self, *args, **kwargs):

        if self.evaluation_method == "cross_validation":
            return self.__cross_validation(*args, **kwargs)
        else:
            return self.__default(*args, **kwargs)
