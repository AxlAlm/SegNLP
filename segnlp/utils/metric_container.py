

#basics
import pandas as pd
from typing import Callable, Union

#segnlp
from segnlp import metrics
from .batch import Batch


class MetricContainer(dict):

    def __init__(self, 
                metric:Union[Callable,str], 
                task_labels: dict,
                ):

        self.task_labels = task_labels

        if isinstance(metric, str):
            metric = getattr(metrics, metric)

        self._metric_fn = metric

        for k in ["train", "val", "test"]:
            self[k] = []


    def calc_add(self, batch: Batch, split:str):
        """
        metrics = {"metic1": 0.3, "metric2": 0.5, ...., "metricn": 0.9 }
        """
        metrics = self._metric_fn(
                                    target_df = batch._df.copy(deep=True), 
                                    pred_df = batch._pred_df.copy(deep=True),
                                    task_labels = self.task_labels
                                    )
        self[split].append(metrics)

        
    def calc_epoch_metrics(self, split:str):
        epoch_metrics = pd.DataFrame(self[split]).mean()
        
        if epoch_metrics.shape[0] == 0:
            return {}

        #epoch_metrics.index = split + "_" + epoch_metrics.index

        self[split] = []
        return epoch_metrics.to_dict()
   
