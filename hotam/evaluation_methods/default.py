


#hotam
from hotam.preprocessing.dataset_preprocessor import PreProcessedDataset
from hotam.ptl import PTLBase
from hotam.ptl import setup_ptl_trainer


def default(
            experiment_id:dict, 
            ptl_trn_args:dict, 
            model_args:dict,
            hyperparamaters:dict, 
            dataset:PreProcessedDataset,
            model_dump_path:str,
            save_choice:str, 
            ):


    dataset.split_id = 0

    trainer = setup_ptl_trainer( 
                                ptl_trn_args=ptl_trn_args,
                                hyperparamaters=hyperparamater, 
                                model_dump_path=exp_dump_path,
                                save_choice=save, 
                                prefix=experiment_id,
                                )

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
    

