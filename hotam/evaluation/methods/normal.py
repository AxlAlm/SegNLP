
def normal(    self, 
                    trainer:Trainer, 
                    ptl_model:PTLBase,
                    dataset:DataSet,
                    save_model_choice:str,
                    run_test:bool
                    ):

        # self.dataset.split_id = 0
        # plt_model = PTLBase(   
        #                     model = model,
        #                     dataset=self.dataset, 
        #                     hyperparamaters=hyperparamaters,
        #                     metrics=metrics,
        #                     monitor_metric=monitor_metric,
        #                     progress_bar_metrics=progress_bar_metrics,
        #                     )

        logger.info("Starting Evalutation (training, validation and testing (optional) ) ... ")
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
        

