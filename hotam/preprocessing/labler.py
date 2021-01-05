
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

logger = get_logger("LABLER")


class Labler:

    """
    class for labeling the heirarchially structured data created from Preprocessor.
    """
    
    def __label(self, row:dict, char_spans:RangeDict) -> dict:
        """label a token unit with label and with BIO

        Parameters
        ----------
        row : dict
            token dataframe row to be labled
        char_spans : RangeDict
            labels with character spans as keys 

        Returns
        -------
        dict
            row of a labeled token 
        """

        char_span = char_spans[row["document_id"]]
        label, label_id = char_span[row["char_end"]]
        label = label.copy() # make a seperate obj for each particular token

        if "None" in label_id:

            if self.__prev_BIO == "I":
                self.__prev_BIO = "O"

            label["ac_id"] = None
        
        else:

            if self.__prev_label_id != label_id and self.__prev_BIO != "B":
                
                label["seg"] = "B"
                self.__prev_BIO = "B"
                self.__ac_id += 1

            else:
                label["seg"] = "I"
                self.__prev_BIO = "I"

            label["ac_id"] = self.__ac_id

        self.__prev_label_id = label_id
 
        return label


    def charspan2label(self, sample_labels:List[dict]):
        """label tokens based on labels attached to character spans. E.g. if character 100-150 is attached to a label X
        all tokens that are between will be labled with X.

        Also creates a new dataframe for the spans, just as sentence, paragraph etc dataframes created
        by Preprocessor

        Parameters
        ----------
        sample_labels : List[dict]
            list of dicts containing the labels for all spans of the text
        span_name : str
            name of the span, e.g. are the spans Argument Units, Opinions or Quotes.
        """

        char_spans = dict(sample_labels)
        args = [char_spans]
        self.__prev_BIO = "O"
        self.__ac_id = -1 #global
        self.__prev_label_id = None

        #label tokens
        tqdm.pandas(desc=f"Adding labels from charspans (+BIO encodings)")
        rows = self.level_dfs["token"].progress_apply(self.__label, axis=1,  args=args)
        self.level_dfs["token"] = pd.concat([self.level_dfs["token"], pd.DataFrame(list(rows))], axis=1)
