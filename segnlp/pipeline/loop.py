


class EarlyStopping:

    def __init__(self, patience:int):
        self.patience = patience
        self.prev_score = 0
        self._n_epochs_wo_imp


    def ___call__(self):

        if self.patience is not None:
            score = epoch_metrics[self.monitor_metric]

            if score < prev_score:
                epochs_wo_improvement += 1
            else:
                epochs_wo_improvement = 0
            
            if epochs_wo_improvement
                    

class SaveBest:

    def __init__(self, path_to_models:str, model_id:str):
        self._path_to_model = os.path.join(path_to_models, model_id)
        self.prev_score = 0


    def ___call__(self, model: torch.nn.Module, score:float):

        if score > self.prev_score:
            model.save(self._path_to_model)

 
                    


class Loop:


    def _train_steps(self, model):

        model.train()

        for batch in DATAMODULE("train"):
            
            loss = model(batch)

            # reset the opt grads
            optimizer.zero_grad()

            # calc grads
            loss.backwards()

            # update paramaters
            optimizer.step()
            

    def _val_steps(self, model):

        model.eval()

        with torch.no_grad():

            for batch in DATAMODULE("VAL"):
                model(batch)
            

    def _loop(self, model_class, loop_type = "train"):

        optim = self._setup_optimizer()
        model = model(model_class)

        prev_score = 0
        epochs_wo_improvement = 0

        for epoch in range(n_epochs):

            self._train_steps(model)

            self._val_steps(model)
            
            train_epoch_metrics = self.model.metrics.calc_epoch_metrics("train")
            val_epoch_metrics = self.model.metrics.calc_epoch_metrics("val")

            self.logger.log_epoch(  
                                epoch = epoch,
                                split = "train"
                                model_id = model_id,
                                epoch_metrics = train_epoch_metrics,
                                )

            self.logger.log_epoch(  
                                epoch = epoch,
                                split = "val"
                                model_id = model_id,
                                epoch_metrics = val_epoch_metrics,
                                )


            score = val_epoch_metrics[self.monitor_metric]

            SaveBest()(model, score)

            if EarlyStopping()(score):
                break

