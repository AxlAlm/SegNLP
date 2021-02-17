
#basics
import re
import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm


#hotam
from hotam.utils import RangeDict
from hotam.utils import timer
from hotam import get_logger

logger = get_logger("LABELER")


class Labeler:

    """
    class for labeling the heirarchially structured data created from Preprocessor.
    """
    
    # def __label(self, row:dict, char_spans:RangeDict) -> dict:
    #     """label a token unit with label and with BIO

    #     Parameters
    #     ----------
    #     row : dict
    #         token dataframe row to be labled
    #     char_spans : RangeDict
    #         labels with character spans as keys 

    #     Returns
    #     -------
    #     dict
    #         row of a labeled token 
    #     """

    #     char_span = char_spans[row["document_id"]]
    #     label, label_id = char_span[row["char_end"]]

    #     label = label.copy() # make a seperate obj for each particular token
    #     #label["seg"] = "O"

    #     if "None" in label_id:

    #         if self.__prev_BIO == "I":
    #             self.__prev_BIO = "O"

    #         label["ac_id"] = None
        
    #     else:

    #         if self.__prev_label_id != label_id and self.__prev_BIO != "B":
                
    #             label["seg"] = "B"
    #             self.__prev_BIO = "B"
    #             self.__ac_id += 1

    #         else:
    #             label["seg"] = "I"
    #             self.__prev_BIO = "I"

    #         label["ac_id"] = self.__ac_id

    #     self.__prev_label_id = label_id
 
    #     return label


    def _label_spans(self, sample:pd.DataFrame, span_labels:dict):

        sample["ac_id"] = np.nan
        rows = self.data.apply("", axis=1,  args=args)
        self.data = pd.concat([self.data, pd.DataFrame(list(rows))], axis=1)



    def _label_spans(self):
        pass

    def _label_tokens(self):
        pass