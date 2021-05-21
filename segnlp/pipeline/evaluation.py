#basics
import numpy as np
from copy import deepcopy

#pytorch
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

#segnlp
from segnlp.utils import get_ptl_trainer_args
from segnlp.utils import DataModule

#pytorch lightning
from pytorch_lightning import Trainer 



class Evaluation:


    def __cross_validation(self,
                            model_args:dict,
                            ptl_trn_args:dict,
                            data_module:DataModule,
                            save_choice:str
                            ):

        cv_scores = []
        model_fps = []
        for i in data_module.splits.keys():
            data_module.split_id = i

            trainer = Trainer(**ptl_trn_args)
            
            checkpoint_cb = None
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    new_filename = callback.filename + f"_fold={i}"
                    setattr(callback, 'filename', new_filename)
                    checkpoint_cb = callback

            model = self.model(**model_args)
            trainer.fit(    
                        model=model, 
                        train_dataloader=data_module.train_dataloader(), 
                        val_dataloaders=data_module.val_dataloader()
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
                save_choice:str,
                ):


        data_module.split_id = 0

        trainer = Trainer(**ptl_trn_args)
        model = self.model(**model_args)
        trainer.fit(    
                        model=model, 
                        train_dataloader=data_module.train_dataloader(), 
                        val_dataloaders=data_module.val_dataloader()
                        )

        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                model_fp = callback.best_model_path
                model_score = float(callback.best_model_score)
            
        return model_fp, model_score


    def _eval(*args, **kwargs):

        if self.evaluation_method == "cross_validation":
            self.__cross_validation(*args, **kwargs)
        else:
            self.__default(*args, **kwargs)

