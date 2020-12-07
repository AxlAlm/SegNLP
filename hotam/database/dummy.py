

import shelve
import os
import pandas as pd


class DummyDB:

    def __init__(self):
        # Database
        self.db = shelve.open("/tmp/hotam_dummy_shelve")
        self.db["experiments"] = []
        self.db["scores"] = []
        self.experiments = self.db["experiments"]
        self.scores = self.db["scores"]
        self.outputs = self.db["outputs"]
    
    
    def close(self):
        self.db.close()

    
    def get_exp(self, experiment_id):
        print("GET EXP", self.get_last_exp())
        return self.get_last_exp()
    

    def get_last_exp(self):
        try:
            return self.experiments[0]
        except IndexError as e:
            return {}


    def get_dropdown_options(self, key, filter_by={}):
        if len(self.experiments):
            unique_items = [self.experiments[0][key]]
        else:
            unique_items = [None]
        return [{'label': e, 'value': e} for e in unique_items]
    

    def get_latest_epoch(self, filter_by):
        try:
            return self.scores[-1]["epoch"]
        except IndexError as e:
            return 0
    

    def get_scores(self,filter_by):
        return self.scores


    def get_experiments(self, filter_by):
        return self.experiments


    def get_metrics(self, filter_by):
        exp = self.get_last_exp()
        return [{"label":m, "value":m} for m in exp.get("metrics",[])]
    

    def get_tasks(self, filter_by):
        exp = self.get_last_exp()
        return [{"label":t, "value":t} for t in exp.get("task2label",{}).keys() ]


    def get_task_classes(self, filter_by, tasks=[]):
        exp = self.get_last_exp()

        task_classes = []
        for task,classes in exp.get("task2label", {}).items():

            if task not in tasks and tasks != []:
                continue
            
            for c in classes:
                task_classes.append({"label":c, "value":c})

        return task_classes