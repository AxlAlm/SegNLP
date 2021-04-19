#basics
import numpy as np


#segnlp
from segnlp.preprocessing.dataset_preprocessor import PreProcessedDataset
from segnlp.ptl import PTLBase
from segnlp.ptl import setup_ptl_trainer

#pytorch lightning
from pytorch_lightning import Trainer 

def cross_validation(
                    model_args:dict,
                    trainer:Trainer,
                    dataset:PreProcessedDataset,
                    save_choice:str
                    ):



    cv_scores = []
    model_fps = []
    for i, ids in self.splits.items():
        
        cp_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                new_filename = callback.filename + f"_fold={i}"
                setattr(model_ckpt_callback, 'filename', new_filename)
                cp_callback = callback

        dataset.split_id = i

        ptl_model = PTLBase(**model_args)
        trainer.fit(    
                    model=ptl_model, 
                    train_dataloader=dataset.train_dataloader(), 
                    val_dataloaders=dataset.val_dataloader()
                    )

        if save_choice == "last":
            model_fp = callback.last_model_path
            checkpoint_dict = torch.load(model_fp)
            model_score = float(checkpoint_dict["callbacks"][ModelCheckpoint]["current_score"])
        else:
            model_fp = callback.best_model_path
            model_score = float(checkpoint_cb.best_model_score)

        cv_scores.append(model_score)
        model_fps.append(model_fp)
    
    mean_score = np.mean(cv_scores)

    return model_fps, mean_score
        