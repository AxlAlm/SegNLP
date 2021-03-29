

import pandas as pd
from collections import Counter
from pprint import pprint


class DataSet:

    def __init__(self, 
                name:str,
                tasks:list,
                task_labels:dict,
                level:str,
                supported_tasks:list,
                supported_prediction_levels:list,
                supported_sample_levels:list,
                prediction_level:str,
                sample_level:str,
                input_level:str,
                about:str="",
                url:str="",
                dump_path:str="/tmp/",
                label_remapping:dict={},
                ):

        self._name = name
        self._tasks = tasks
        self._task_labels = task_labels
        self._level = input_level
        self._supported_tasks = supported_tasks
        self._supported_prediction_levels = supported_prediction_levels
        self._supported_sample_levels = supported_sample_levels
        self._prediction_level=prediction_level
        self._sample_level=sample_level
        self.dump_path = dump_path

        # check for duplicates in tasks
        unpacked_tasks = set()
        for task in tasks:
            for st in tasks.split("+"):
                if st in unpacked_tasks:
                    raise RuntimeError(f"{st} found in more than one task")
                else:
                    st.add(st)
        
        assert prediction_level in self._supported_prediction_levels
        assert sample_level in self._supported_sample_levels
        assert set(subtasks).issubset(set([self._supported_tasks]))
    
        self._dataset_path = self._download_data()
        self.data = self._process_data()
        self._splits = self._splits()


    def __getitem__(self,key):
        return self.data.loc[key].to_dict("record")


    def __len__(self):
        return self.data.shape[0]
    

    @property
    def level(self):
        return self._level

    @property
    def sample_level(self):
        return self._sample_level 

    @property
    def prediction_level(self):
        return self._prediction_level

    @property
    def tasks(self):
        return self._tasks 

    @property
    def task_labels(self):
        return self._task_labels

    @property
    def splits(self):
        return self._splits


    def _download_data(self):
        raise NotImplementedError
 

     def _process_data(self):
        raise NotImplementedError
 

     def _splits(self):
        raise NotImplementedError
 
 
    # @classmethod
    # def load_CoNLL(self):
    #     raise NotImplementedError


    # @classmethod
    # def load_DAT(self):
    #     raise NotImplementedError


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
        
        return {t:dict(c) for t,c in collected_counts.items()}


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
