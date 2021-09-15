
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
