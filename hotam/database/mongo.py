

import pymongo

import os
import pandas as pd


class MongoDB:


    def __init__(self):
        # Database
        if 'MONGO_KEY' in os.environ:
            client = pymongo.MongoClient(os.environ['MONGO_KEY'])
            self.db = client[os.environ['DB_NAME']]
        else:
            client = pymongo.MongoClient()
            self.db = client["dummy_db"]

        self.experiments = self.db["experiments"]
        self.outputs = self.db["outputs"]
        self.scores = self.db["scores"]

    
    def get_last_exp(self):
        last_added_exp = self.experiments.find_one(sort=[('start_timestamp', pymongo.DESCENDING)])
        return last_added_exp if last_added_exp else {}


    def get_dropdown_options(self, key, filter_by={}):
        unique_items = self.experiments.distinct(key, filter=filter_by)
        return [{'label': e, 'value': e} for e in unique_items]
    

    def get_last_epoch(self, filter_by):
        try:
            return self.scores.find_one(filter_by, sort=[('epoch', pymongo.DESCENDING)])["epoch"]
        except TypeError:
            return -1


    def get_scores(self, filter_by):
        if "experiment_id" not in filter_by:
            return pd.DataFrame([])

        return pd.DataFrame(list(self.scores.find(filter_by)))
    

    def get_outputs(self,filter_by):

        if "experiment_id" not in filter_by:
            return {}

        output =  self.outputs.find_one(filter_by)
        return {} if output is None else output
    

    def get_exp_config(self, filter_by):
        return self.experiments.find_one(filter_by)


    def get_exp_configs(self, filter_by):
        return list(self.experiments.find(filter_by))


    def get_metrics(self, filter_by, many=True):
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

    def get_live_exps_ids(self):
        exp_ids = [exp["experiment_id"] for exp in list(self.experiments.find({"status":"ongoing"}))]
        return exp_ids

    def get_done_exps_ids(self):
        exp_ids = [exp["experiment_id"] for exp in list(self.experiments.find({"status":"done"}))]
        return exp_ids

    def get_projects(self):
        projects = sorted(set([exp["project"] for exp in list(self.experiments.find({"status":"done"}))]))
        return projects
    
    def get_project_tasks(self, project):
        tasks = sorted(set([t for exp in list(self.experiments.find({"project":project})) for t in exp["tasks"]]))
        return tasks
    
    def get_project_subtasks(self, project):
        subtasks = sorted(set([t for exp in list(self.experiments.find({"project":project})) for t in exp["subtasks"]]))
        return subtasks