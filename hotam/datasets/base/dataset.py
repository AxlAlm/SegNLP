

import pandas as pd
from collections import Counter
from pprint import pprint

class DataSet:


    def __getitem__(self,key):
        if isinstance(key, int):
            return self.data.iloc[key].to_dict()
        else:
            return self.data.loc[key].to_dict("record")

    def __len__(self):
        return self._size
    
    @property
    def tasks(self):
        return self._tasks 

    @property
    def task_labels(self):
        return self._task_labels

    @property
    def splits(self):
        return self._splits


    @classmethod
    def load_CoNLL(self):
        raise NotImplementedError


    @classmethod
    def load_DAT(self):
        raise NotImplementedError


    def stats(self):

        collected_counts = {task:Counter() for task in self.tasks}

        if "span_labels" in self.data.columns:
            for i, row in self.data.iterrows():
                span_labels = row["span_labels"]
                sdf = pd.DataFrame(list(span_labels.values()))
                sdf = sdf[~sdf["unit_id"].isna()]

                for task in self.tasks:
                    counts = sdf[task].value_counts().to_dict()
                    collected_counts[task] += counts
        
        pprint({t:dict(c) for t,c in collected_counts.items()})


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
