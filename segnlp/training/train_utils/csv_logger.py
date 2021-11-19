
#basics
import os
from glob import glob
from random import random
import pandas as pd
import re



class CSVLogger:

    def __init__(self, path_to_logs:str) -> None:
        self._path_to_logs = path_to_logs
        os.makedirs(self._path_to_logs, exist_ok = True)


    def log(self, data:dict) -> None: 

        is_first = data["epoch"] == 0

        if is_first:
        
            #create log file
            self._log_file = os.path.join(self._path_to_logs, f'{data["split"]}.log')

            # create the file
            open(self._log_file, "w")


        with open(self._log_file, "a") as f:
            
            # create a header if file is new
            if is_first:
                f.write(",".join(list(data.keys())) + "\n")

            # row
            f.write(",".join([str(v) for v in data.values()]) + "\n")

        
     
