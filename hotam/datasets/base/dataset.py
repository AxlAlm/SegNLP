


class DataSet:

    def __getitem__(self,key):
        return self.data[key]


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


    def stats(self):
        rows.append({
                        "type":self.sample_level,
                        "task":"",
                        "split":split_type,
                        "split_id":split_id,
                        "value": len(ids)

                    })

    def info(self):
        doc = f"""
            Info:

            {self.about}

            Tasks: 
            {self.tasks}

            Task Labels:
            {self.task_labels}

            Source:
            {self.url}
            """
        print(doc)
