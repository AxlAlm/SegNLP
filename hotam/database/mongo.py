

import pymongo

import os
import pandas as pd


class MongoDataBase:


    def __init__(self):
        # Database
        client = pymongo.MongoClient(os.environ['MONGO_KEY'])
        db = client[os.environ['DB_NAME']]
        self.experiments = db["experiments"]
        self.outputs = db["outputs"]
        self.scores = db["scores"]
        self.datasets = db["datasets"]

    
    def get_last_exp(self):
        last_added_exp = self.experiments.find_one(sort=[('start_timestamp', pymongo.DESCENDING)])
        return last_added_exp if last_added_exp else {}


    def get_dropdown_options(self, key, filter_by={}):
        unique_items = self.experiments.distinct(key, filter=filter_by)
        return [{'label': e, 'value': e} for e in unique_items]
    

    def get_latest_epoch(self, filter_by):
        #result = self.scores.find_one(filter_by, sort=[('epoch', pymongo.DESCENDING)]))
        return self.scores.find_one(filter_by, sort=[('epoch', pymongo.DESCENDING)])["epoch"]


    def get_scores(self,filter_by):
        
        if "experiment_id" not in filter_by:
            return pd.DataFrame([])

        return pd.DataFrame(list(self.scores.find(filter_by)))


    def get_experiments(self, filter_by):
        return list(self.experiments.find(filter_by))


    def get_metrics(self, filter_by):
        exp = self.experiments.find_one(filter_by)
        return [{"label":m, "value":m} for m in exp["metrics"]]
    

    def get_tasks(self, filter_by):
        exp = self.experiments.find_one(filter_by)
        return [{"label":t, "value":t} for t in list(exp["task2label"].keys())]


    def get_task_classes(self, filter_by, tasks=[]):
        exp = self.experiments.find_one(filter_by)

        task_classes = []
        for task,classes in exp["task2label"].items():

            if task not in tasks and tasks != []:
                continue
            
            for c in classes:
                task_classes.append({"label":c, "value":c})

        return task_classes