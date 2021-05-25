

#basics
import pandas as pd
from typing import Callable, Union

#segnlp
from segnlp import metrics

class MetricContainer(dict):

    def __init__(self, metric=Union[Callable,str]):

        if isinstance(metric, str):
            metric = getattr(metrics, metric)

        self._metric_fn = metric
       
        for k in ["train", "val", "test"]:
            self[k] = []


    def calc_add(self, output:pd.DataFrame, split:str):
        """
        metrics = {"metic1": 0.3, "metric2": 0.5, ...., "metricn": 0.9 }
        """
        metrics = self._metric_fn(output)
        self[split].append(metrics)

        
    def calc_epoch_metrics(self, split:str):
        #print(pd.DataFrame(self[split]))
        #print(pd.DataFrame(self[split]).mean())
        epoch_metrics = pd.DataFrame(self[split]).mean()
        
        if epoch_metrics.shape[0] == 0:
            return {}

        epoch_metrics.index = split + "_" + epoch_metrics.index
        return epoch_metrics.to_dict()
   
