
                 
class Loop:


    def _train_steps(self,
                     model: nn.Module, 
                     optimizer : nn.otpim,
                     gradient_clip_val : float = None
                     ) -> None:

        model.train()

        # tqdm(range(n_epochs), start=0, desc = "Train Steps", position=3)
        for batch in DATAMODULE("train"):
            
            loss = model(batch)

            # reset the opt grads
            optimizer.zero_grad()

            # calc grads
            loss.backwards()

            #clip gradients
            if gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            # update paramaters
            optimizer.step()


        train_epoch_metrics = self.model.metrics.calc_epoch_metrics("train")
        self.logger.log_epoch(  
                            epoch = epoch,
                            split = "train"
                            model_id = model_id,
                            epoch_metrics = train_epoch_metrics,
                            cv = 0
                            )


    def _val_steps(self, model : nn.Module) -> None:

        model.eval()

        with torch.no_grad():

            tqdm(range(n_epochs), start=0, desc = "Train Steps", position=3)
            for batch in DATAMODULE("VAL"):
                model(batch)
         
        val_epoch_metrics = self.model.metrics.calc_epoch_metrics("val")

        self.logger.log_epoch(  
                            epoch = epoch,
                            split = "val"
                            model_id = model_id,
                            epoch_metrics = val_epoch_metrics,
                            cv = 0
                            )


    def _loop(self, 
                model_class
                cv : int = 0
                device = None,
                ):

        #loading our preprocessed dataset
        data_module = utils.DataModule(
                                path_to_data = self._path_to_data,
                                batch_size = hyperparamaters["general"]["batch_size"],
                                cv = cv
                                )

        #setup learning scheduler
        optimizer, lr_scheduler = self._configure_optimizers(hyperparamters)


        #set up model
        model = model(
                        model_class
                        )

        if gpu:
            model = model.to(device)
                        

        for epoch in tqdm(range(n_epochs), start=0, desc = "Epochs", position=2):
            
            if self.training:
                self._train_steps(model, optimizer)
                self._val_steps(model)
            
                score = val_epoch_metrics[self.monitor_metric]

                SaveBest()(model, score)

                if EarlyStopping()(score):
                    break

                if lr_scheduler is not None:
                    lr_scheduler.step()

            else:
                self._val_steps(model)

    

    def _cv_loop(self):

        for i in range(n_cvs):
            self._loop()
            


    def fit(self,) -> None:

        model_save_path = os.path.join(self._path_to_models, model_id, str(random_seed))


        if CV:
            self._cv_loop()

        else:
             self._loop()