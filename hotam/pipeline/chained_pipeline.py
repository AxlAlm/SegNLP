
#basic
#from __future__ import annotations
import pandas as pd
import os
from glob import glob
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Union
import itertools
import json
import warnings
from IPython.display import display
import numpy as np
import random
from copy import deepcopy
#import webbrowser
import time
import sys

#hotam
from hotam import get_logger
from hotam.pipeline import Pipeline

#warnings.simplefilter('ignore')


class ChainedPipeline:


    def __init__(self, pipelines=Union[List, Pipeline]):
        self.pipelines = pipelines


    def fit(*args, **kwargs):
        for pipeline in self.pipelines:
            pipeline.fit(*args, **kwargs)
                    

    def eval(self):
        for pipeline in self.pipelines:
            pipeline.eval()


    def test(self):
        pass
        # outputs = []
        # for pipeline in self.pipelines:

        #     output = self.pipelines.test(

        #                                 overrride_label_df = output[-1]
        #                                 )
        #     outputs.append(output)

        # return outputs


    def predict(self, doc:Union[str,List[str]], chain_predictions:bool = True):
        pass
        # for pipeline in self.pipelines:
        #     out = pipeline.predict(doc)

        #     #overwrite the doc so input to the next pipeline is output of the previuos pipeline
        #     if chain_predictions:
        #         doc = out
        
        # #return output of last pipeline
        # return out
        