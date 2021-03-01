



class LocalDB:

    """
    structure

    root_dir/
        experiments/
            <exp_id>_<project>_<dataset>_<model>_<timestamp>/
                config.json
                scores/
                    val/
                        epoch=1.json
                        epoch=n.json
                        ..
                    train/
                        epoch=n.json
                outputs/
                    epoch=1.json
                    epoch=2.json
                    ..


    """





import pymongo

import os
from glob import glob
import pandas as pd

def read_json(fp):
    with open(fp, "r") as f:
        content = json.load(f)
    return content

class MongoDB:


    def __init__(self):
        self.root_dir = root_dir
        self.experiment_dir = os.path.join(self.root_dir, "experiments")
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)


    def get_last_exp(self):
        all_exp_folders = glob(self.experiment_dir+"/*")
        last_exp_folder = sorted(all_exp_folders, lambda x: int(x.split("_"[-1])))[-1]
        last_exp_config = os.path.join(last_exp_folder, "config.json")
        with open(last_exp_config, "r") as f:
            config = json.load(f)

        return config

    def get_last_epoch(self, experiment_id):
        try:
            exp_folder = glob(os.path.join(self.experiment_dir, experiment_id+"*"))[0]

            train_scores  = os.path.join(exp_folder, "scores", "train")
            val_scores  = os.path.join(exp_folder, "scores", "val")
            get_epoch = lambda x:int(re.sub(r"[^\d+]", x, ""))
            last_epoch = max(
                                sorted([get_epoch(x) for x in train_scores])[-1],
                                sorted([get_epoch(x) for x in train_scores])[-1]
                                )

            return last_epoch
        except IndexError:
            return -1


    def get_scores(self, experiment_ids:list, epoch:int = None):
        
        scores = []
        #make this into a regex search instead?
        for exp_id in experiment_ids:
            score_files = glob(os.path.join(self.experiment_dir, f"{exp_id}*/scores**.json"))
            scores.extend(read_json(fp) for fp in score_files)

        return pd.DataFrame(scores)
    

    def get_outputs(self, experiment_ids):

        outputs = []
        #make this into a regex search instead?
        for exp_id in experiment_ids:
            output_files = glob(os.path.join(self.experiment_dir, f"{exp_id}*/outputs**.json"))
            outputs.extend(read_json(fp) for fp in output_files)

        return {} if output is None else output
    

    def get_exp_config(self, experiment_ids):
        config_file = glob(os.path.join(self.experiment_dir, experiment_id+"*/config.json"))[0]
        config = read_json(config_file)
        return config


    def get_exp_configs(self, filter_by):
        return list(self.experiments.find(filter_by))


    def get_metrics(self, filter_by, many=True):
        exp = self.experiments.find_one(filter_by)
        return [{"label":m, "value":m} for m in exp["metrics"]]
    

    def get_tasks(self, filter_by):
        exp = self.experiments.find_one(filter_by)
        return [{"label":t, "value":t} for t in exp["tasks"]]

    def get_subtasks(self, filter_by):
        exp = self.experiments.find_one(filter_by)
        return [{"label":t, "value":t} for t in exp["subtasks"]]

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