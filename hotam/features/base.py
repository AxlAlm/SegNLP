
# basic
import numpy as np
from abc import ABC, abstractmethod
import os
from typing import List, Dict, Tuple
import re
import copy
from pathlib import Path


# axl nlp
import hotam
import hotam.utils as u
from hotam.features import get_feature
from hotam.preprocessing import DataSet
from hotam import get_logger



logger = get_logger("FEATURES")


def feature_memory(extract):


    def load(class_object, df, id_):
        #logger.info("loading")
        return copy.deepcopy(class_object._mem_data[id_])


    def save(class_object, df, id_):
        #logger.info("saving")
        f_emb = extract(class_object, df)

        if len(f_emb.shape) > 1:
            length = f_emb.shape[0]
            class_object._mem_data[id_][:length] = f_emb
        else:
            class_object._mem_data[id_] = f_emb
 
        return f_emb


    def wrapper(*args):
        class_object = args[0]
        df = args[1]
        id_ = int(df.index.max())

        if hotam.preprocessing.settings["STORE_FEATURES"]:
            return extract(class_object, df)

        # if we havent saved all features yet
        if not class_object._all_saved():
            out = save(class_object, df, id_)
            class_object._saved_ids.add(id_)
        else:
            out = load(class_object, df, id_)
        
        return out

    return wrapper


class FeatureModel(ABC):


    def _init_feature_save(self, dataset_name:str, feature_name:str, shape:tuple, dtype:str): # memmap_file:str,

        dir_path = f"{str(Path.home())}/.hotam/features/{dataset_name}"
        os.makedirs(dir_path, exist_ok=True)

        # to keep track on which ids we have saved
        # maybe move this id column memmap??
        self.__idsets_path = os.path.join(dir_path, f"{feature_name}_ids.json")
        ids_exist = os.path.exists(self.__idsets_path)
        if ids_exist:
            self._load_ids()
        else:
            self.__saved_ids = set()

        # the memmap
        self.__memmap_path = os.path.join(dir_path, f"{feature_name}.dat")
        memmap_exists = os.path.exists(self.__memmap_path)
        if memmap_exists:
            mode = "r+"
        else:
            mode = "w+"
    
        self._mem_data  = np.memmap(
                                        self.__memmap_path, 
                                        dtype=dtype, 
                                        mode=mode, 
                                        shape=shape
                                        )
    

    def _dump_ids(self):
        with open(self.__idsets_path, "w") as f:
            json.dump(list(self.__saved_ids), f, indent=4)


    def _load_ids(self):
        with open(self.__idsets_path, "r") as f:
            self.__saved_ids = set(json.load(f))

    
    def _all_saved(self):

        if len(self.__saved_ids) == self._mem_data.shape[0]-1:
            #logger.info("All Features are saved")

            if callable(getattr(self, "deactivate", None)):
                self.deactivate()
            
            self._dump_ids()
            return True

        else:
            return False
        
        
    @property
    def name(self):
        return self._name

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def level(self):
        return self._level

    @property
    def dtype(self):
        return self._dtype

    
    @property
    def context(self):
        if hasattr(self, "_context"):
           return self._context
        else:
            return False
    

    @abstractmethod
    def extract(self):
        pass

