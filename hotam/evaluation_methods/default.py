


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

    dataset.split_id = 0

    ptl_model = PTLBase(**model_params)
    trainer.fit(    
                    model=ptl_model, 
                    train_dataloader=dataset.train_dataloader(), 
                    val_dataloaders=dataset.val_dataloader()
                    )

    #picking model to run on test split
    if save_choice is not None:
        
        if save_choice == "last":
            trainer.test(
                        model=ptl_model, 
                        test_dataloaders=dataset.test_dataloader()
                        )
        elif save_choice == "best":
            trainer.test(
                        model="best",
                        test_dataloaders=dataset.test_dataloader()
                        )
        else:
            raise RuntimeError(f"'{save_choice}' is not an approptiate choice when testing models")
    

