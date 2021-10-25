
#basics
import os
from glob import glob
from random import random
import pandas as pd
import re



class CSVLogger:

    def __init__(self, log_file:str) -> None:

        #create log file
        self._log_file = log_file

        # create the file
        open(self._log_file, "w")

        # set headers on the first call
        self.first : bool = True

    
    def log(self, data:dict) -> None:                            

        with open(self._log_file, "a") as f:
            
            # create a header if file is new
            if self.first:
                f.write(",".join(list(data.keys())) + "\n")
                self.first = False

            # row
            f.write(",".join([str(v) for v in data.values()]) + "\n")

        
     
