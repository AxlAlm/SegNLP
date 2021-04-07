#hotam
from hotam.preprocessing.dataset_preprocessor import PreProcessedDataset
from hotam.ptl import PTLBase
from hotam.ptl import setup_ptl_trainer

#pytorch lightning
from pytorch_lightning import Trainer 

def cross_validation(
                    model_args:dict,
                    trainer:Trainer,
                    dataset:PreProcessedDataset,
                    ):



    for i, ids in self.splits.items():

        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                new_filename = callback.filename + f"_fold={i}"
                setattr(model_ckpt_callback, 'filename', new_filename)

        dataset.split_id = i

        ptl_model = PTLBase(**model_args)
        trainer.fit(    
                    model=ptl_model, 
                    train_dataloader=dataset.train_dataloader(), 
                    val_dataloaders=dataset.val_dataloader()
                    )