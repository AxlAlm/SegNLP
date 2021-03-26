
from hotam.datasets.am.pe import  PE
from hotam.datasets.base import DataSet

def get_dataset(dataset_name):
    if dataset_name.lower() == "pe":
        return PE


__all__ = [
            "DataSet",
            "PE",
            ]
