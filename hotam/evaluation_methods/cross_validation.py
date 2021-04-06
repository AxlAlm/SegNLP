


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


    for i, ids in self.splits.items():

        dataset.split_id = i
        trainer = setup_ptl_trainer( 
                                    ptl_trn_args=ptl_trn_args,
                                    hyperparamaters=hyperparamaters, 
                                    model_dump_path=model_dump_path,
                                    save_choice=save_choice, 
                                    prefix=experiment_id+"_cv={i}",
                                    )

        ptl_model = PTLBase(**model_args)
        trainer.fit(    
                    model=ptl_model, 
                    train_dataloader=dataset.train_dataloader(), 
                    val_dataloaders=dataset.val_dataloader()
                    )