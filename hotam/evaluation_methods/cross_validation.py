


#hotam
from hotam.preprocessing.dataset_preprocessor import PreProcessedDataset
from hotam.ptl import PTLBase
from pytorch_lightning import Trainer


def default( 
            trainer:Trainer, 
            model_params:dict,
            dataset:PreProcessedDataset,
            save_choice:str,
            ):


    for i, ids in self.splits.items():
        dataset.split_id = i

        ptl_model = PTLBase(**model_params)
        trainer.fit(    
                    model=ptl_model, 
                    train_dataloader=dataset.train_dataloader(), 
                    val_dataloaders=dataset.val_dataloader()
                    )