
from segnlp.datasets.am.pe import  PE
from segnlp.datasets.am.mtc import MTC

from segnlp.datasets.base import DataSet

def get_dataset(dataset_name):
    if dataset_name.lower() == "pe":
        return PE

__all__ = [
            "DataSet",
            "PE",
            "MTC",
            ]
