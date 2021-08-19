

#basics
import pandas as pd
from typing import Callable, Union

#segnlp
from segnlp import metrics

class MetricContainer(dict):

    def __init__(self, metric:Union[Callable,str], label_encoders:dict):

        self.task_labels = {task: list(enc.id2label.values()) if task != "link" else [] for task, enc in label_encoders.items() }
        self.task_label_ids = {task: list(enc.id2label.keys()) if task != "link" else [] for task, enc in label_encoders.items() }

        if isinstance(metric, str):
            metric = getattr(metrics, metric)

        self._metric_fn = metric

        for k in ["train", "val", "test"]:
            self[k] = []


    def calc_add(self, df:pd.DataFrame, split:str):
        """
        metrics = {"metic1": 0.3, "metric2": 0.5, ...., "metricn": 0.9 }
        """

        if self._metric_fn.__name__ == "overlap_metric":
            metrics = self._metric_fn(df, self.task_labels)
        else:
            metrics = self._metric_fn(df, self.task_labels, self.task_label_ids)

        self[split].append(metrics)

        
    def calc_epoch_metrics(self, split:str):
        epoch_metrics = pd.DataFrame(self[split]).mean()
        
        if epoch_metrics.shape[0] == 0:
            return {}

        epoch_metrics.index = split + "_" + epoch_metrics.index
        
        self[split] = []
        return epoch_metrics.to_dict()
   
