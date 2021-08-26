
#basics
import pandas as pd
import numpy as np

# segnlp
from segnlp.pretrained_features.base import FeatureModel



class SegPos(FeatureModel):

    """
    Creates a vector representing a sentences position in the text.

    original texts are needed to create this feature

    feature is specifically implemented for replication of Joint Pointer NN from :
    https://arxiv.org/pdf/1612.08994.pdf

    "(3) Structural features:  Whether or not the AC is the first AC in a paragraph, 
    and Whether the AC  is  in  an  opening,  body,  or  closing  paragraph."

    we represent this info as a one hot encodings of dim==4
    """

    def __init__(self, group:str="doc_embs"):
        self._feature_dim = 4
        self._name = "docpos"
        self._level = "doc"
        self._group = self._name if group is None else group
        self._dtype = np.uint8

        self.__para_id = -1
        self.__seg_id = 0

    #@feature_memory
    def extract(self, df):
        """
        extracts document position for paragraphs. 

        one hot encodig where dimN == {0,1}, dim size = 4
        dim0 = 1 if item is first in sample 0 if not
        dim1 = 1 if item is in first paragraph if not
        dim2 = 1 if item is in body 0 if not
        dim3 = 1 if item is in last paragraph 0 if not

        Parameters
        ----------
        sample_ids : list, optional
            sample ids for which to extract features, by default None
        pad : bool, optional
            if pad or not, by default True

        Returns
        -------
        np.ndarray
            matrix (padded or not)

        Raises
        ------
        ValueError
            sample level has to be document for this feature
        """
        vec = np.zeros(4)

        # if the para_id of the seg  is different then the previous para_id seen,
        # this means that we have changed paragraph, hence we are at the first seg in a new paragraph
        para_id = df[f"paragraph_id"].max()
        if para_id != self.__para_id:
            vec[0] = 1
            self.__para_id = para_id

        doc_para_id = df[f"document_paragraph_id"].max()
        last_para = df[f"nr_paragraphs_doc"].max() - 1 

        # if seg is in the first paragraph
        if doc_para_id == 0:
            vec[1] = 1

        # if seg is in the body
        elif doc_para_id > 0 and doc_para_id < last_para:
            vec[2] = 1

        # if its not in the first or in the body, its in the last
        else:
            vec[3] = 1
        
        return vec