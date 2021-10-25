


class EarlyStopping:

    def __init__(self, patience:int):
        self._patience = patience
        self._top_score = 0
        self._n_epochs_wo_improvement = 0


    def __call__(self, score) -> bool:

        if self._patience is None:
            return False

        if score < self._top_score:
            self._n_epochs_wo_improvement += 1
        else:
            self._top_score = score
            self._n_epochs_wo_improvement = 0

        if self._n_epochs_wo_improvement > self._patience:
            return True
   
