


class DataSet:


    def __len__(self):
        return self._size
    
    @property
    def name(self):
        return self._name

    @property
    def task(self):
        return self._tasks 
    
    @property
    def task_labels(self):
        return self._task_labels

    @property
    def splits(self):
        return self._splits

    def stats(self):
        raise NotImplementedError


    @classmethod
    def load_CoNLL(self):
        raise NotImplementedError


    @classmethod
    def load_DAT(self):
        raise NotImplementedError