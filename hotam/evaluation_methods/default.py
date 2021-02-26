


#hotam
from hotam.preprocessing.dataset_preprocessor import PreProcessedDataset
from hotam.ptl import PTLBase
from pytorch_lightning import Trainer


def default( 
            trainer:Trainer, 
            ptl_model:PTLBase,
            dataset:PreProcessedDataset,
            save_choice:str,
            ):

        dataset.split_id = 0
        trainer.fit(    
                        model=ptl_model, 
                        train_dataloader=dataset.train_dataloader(), 
                        val_dataloaders=dataset.val_dataloader()
                        )

        if test_set is not None:
            
            if save_choice == "last":
                trainer.test(
                            model=ptl_model, 
                            test_dataloaders=self.dataset.test_dataloader()
                            )
            elif save_choice == "best":
                trainer.test(
                            model="best",
                            test_dataloaders=self.dataset.test_dataloader()
                            )
            else:
                raise RuntimeError(f"'{save_choice}' is not an approptiate choice when testing models")
        

