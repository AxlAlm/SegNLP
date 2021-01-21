#basic
from typing import Tuple, List, Dict, Callable


#hotam
from hotam.manager.ptl_trainer_setup import Trainer
from hotam.trainer.base import PTLBase

#torch
import torch.nn as nn


class Evaluator:

    """ Class for evaluation. Contain all supported evaluation methods
    """

    def _get_eval_method(self) -> Callable:
        """returns evaluation method

        Returns
        -------
        Callable
            function for evaluation method
        """

        if self.evaluation_method == "default":
            return self.default

    
    def default(    self, 
                    trainer:Trainer, 
                    model:nn.Module,
                    hyperparamaters:dict,
                    metrics:None, 
                    monitor_metric:None,
                    progress_bar_metrics:None,
                    save_model_choice:str,
                    run_test:bool
                    ):

        """trains and evaluates a model with given hyperparamaters by:

            trains on train split
            validates on validation split

        Parameters
        ----------
        trainer : Trainer
            Trainer class
        model_class : PyToBase
            Base NN model class
        hyperparams : dict
            hyperperanaters to be used for the model
        save_last : bool


        Returns
        -------
        dict
            logs; scores, predictions
        """

        self.dataset.split_id = 0
        plt_model = PTLBase(   
                            model = model,
                            dataset=self.dataset, 
                            hyperparamaters=hyperparamaters,
                            metrics=metrics,
                            monitor_metric=monitor_metric,
                            progress_bar_metrics=progress_bar_metrics,
                            )

        trainer.fit(    
                        model=plt_model, 
                        train_dataloader=self.dataset.train_dataloader(), 
                        val_dataloaders=self.dataset.val_dataloader()
                        )

        if run_test:
            
            if save_model_choice == "last":
                trainer.test(
                            model=model, 
                            test_dataloaders=self.dataset.test_dataloader()
                            )
            elif save_model_choice == "best":
                trainer.test(
                            model="best",
                            test_dataloaders=self.dataset.test_dataloader()
                            )
            else:
                raise RuntimeError(f"'{save_model_choice}' is not an approptiate choice when testing models")


    def cross_validation(self, trainer:Trainer, model_class:None, hyperparams:dict) -> dict:
        """
        trains and evaluates a model with given hyperparamaters by Cross Validation:

            trains on train split N
            validates on validation split N

        Parameters
        ----------
        trainer : Trainer
            Trainer class
        model_class : PyToBase
            Base NN model class
        hyperparams : dict
            hyperperanaters to be used for the model

        Returns
        -------
        dict
            logs; scores, predictions
        """
        while True:
            model = model_class(self.dataset, self.features, hyperparams)
            trainer.fit(model)

