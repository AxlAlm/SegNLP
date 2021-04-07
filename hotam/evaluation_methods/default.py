#hotam
from hotam.preprocessing.dataset_preprocessor import PreProcessedDataset
from hotam.ptl import PTLBase
from hotam.ptl import setup_ptl_trainer


#pytorch lightning
from pytorch_lightning import Trainer 


def default(
            model_args:dict,
            trainer:Trainer,
            dataset:PreProcessedDataset,
            ):


    dataset.split_id = 0


    ptl_model = PTLBase(**model_args)
    trainer.fit(    
                    model=ptl_model, 
                    train_dataloader=dataset.train_dataloader(), 
                    val_dataloaders=dataset.val_dataloader()
                    )


