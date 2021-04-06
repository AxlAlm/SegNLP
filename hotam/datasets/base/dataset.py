

import pandas as pd
from collections import Counter
from pprint import pprint
import itertools


class DataSet:

    def __init__(self, 
                name:str,
                tasks:list,
                supported_task_labels:dict,
                level:str,
                supported_tasks:list,
                supported_prediction_levels:list,
                supported_sample_levels:list,
                prediction_level:str,
                sample_level:str,
                about:str="",
                url:str="",
                download_url:str="",
                dump_path:str="/tmp/",
                ):

        self._name = name
        self._level = level
        self._supported_task_labels = supported_task_labels
        self._supported_tasks = supported_tasks
        self._supported_prediction_levels = supported_prediction_levels
        self._supported_sample_levels = supported_sample_levels
        self._prediction_level=prediction_level
        self._sample_level=sample_level
        self.dump_path = dump_path

        self.url = url
        self.download_url = download_url
        self.about = about

        self._tasks = tasks
        self._subtasks = self.__get_subtasks(tasks)

        for st in self._subtasks:
            if self._subtasks.count(st) > 1:
                raise RuntimeError(f"'{st}' found in more than one task")

        if not set(self._subtasks).issubset(set(self._supported_tasks)):
            raise RuntimeError("Given list of tasks contain unsupported tasks")

        if prediction_level == "token":
            if "seg" not in self._subtasks:
                raise RuntimeError("if prediction level is 'token', 'seg' must be included among the subtasks")

        if prediction_level not in self._supported_prediction_levels:
            raise RuntimeError(f"'{prediction_level}' is not in supported prediction levels: {self._supported_prediction_levels}")
        
        if sample_level not in self._supported_sample_levels:
            raise RuntimeError(f"'{sample_level}' is not in supported sample levels: {self._supported_sample_levels}")


        self._task_labels = self.__get_task_labels(tasks, supported_task_labels)

        path_to_data = self._download_data()
        self.data = self._process_data(path_to_data)
        self._splits = self._splits()

        assert isinstance(self.data, pd.DataFrame)
        

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
    def subtasks(self):
        return self._subtasks

    @property
    def supported_tasks(self):
        return self._supported_tasks 

    @property
    def supported_task_labels(self):
        return self._supported_task_labels

    @property
    def supported_sample_levels(self):
        return self._supported_sample_levels

    @property
    def supported_prediction_levels(self):
        return self._supported_sample_levels

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


    def __get_subtasks(self, tasks):
        subtasks = []
        for task in tasks:
            subtasks.extend(task.split("+"))
        return subtasks


    def __get_task_labels(self, tasks:list, supported_task_labels:dict):
        
        task_labels = {}
        for task in tasks:

            subtasks = task.split("+")

            if len(subtasks) < 2:
                task_labels[task] = supported_task_labels[task]
                continue

            label_groups = []
            has_seg = False
            for st in subtasks:
                task_labels[st] = supported_task_labels[st]

                if st == "seg":
                    BIO = task_labels["seg"].copy()
                    BIO.remove("O")
                    label_groups.append(BIO)
                    has_seg = True
                else:
                    label_groups.append(task_labels[st])

            combs = list(itertools.product(*label_groups))

            if has_seg:
                none_label = ["O"] + ["None" if s != "link" else "0" for s in task_labels if s != "seg"]
                combs.insert(0,none_label)

            task_labels[task] = ["_".join([str(c) for c in comb]) for comb in combs]
        
        return task_labels


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
