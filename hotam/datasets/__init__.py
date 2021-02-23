
from hotam.datasets.pe import  PE

# def get_dataset(dataset_name:str, dump_path:str="/tmp/"):

#     if dataset_name.lower() == "pe":
#         return PE_Dataset(dump_path=dump_path).load()
#     else:
#         raise KeyError(f'"{dataset_name}" is not a supported dataset')

__all__ = [
            PE,
            
            ]