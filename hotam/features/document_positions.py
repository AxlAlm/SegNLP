
from hotam.features.base import FeatureModel, feature_memory
from hotam.preprocessing import DataSet
import pandas as pd
import numpy as np


class DocPos(FeatureModel):

    """
    Creates a vector representing a sentences position in the text.

    original texts are needed to create this feature

    feature is specifically implemented for replication of Joint Pointer NN from :
    https://arxiv.org/pdf/1612.08994.pdf

    "(3) Structural features:  Whether or not the AC is the first AC in a paragraph, 
    and Whether the AC  is  in  an  opening,  body,  or  closing  paragraph."
    """

    def __init__(self, dataset:DataSet, prediction_level:str):
        self._feature_dim = 2
        self._name = "docpos"

        #create a dict of document 2 number of paragraphs
        para_groups = dataset.level_dfs["paragraph"].groupby("document_id", sort=False)
        self.doc2paralen = {i:g.shape[0] for i,g in para_groups}
        self.prediction_level = prediction_level
        self._level = "doc"
        self._dtype = np.uint8

    #@feature_memory
    def extract(self, df):
        """
        extracts document position for paragraphs. 

        feature dim == 2 :
        # 0) is the first AC in a paragaraph
        # 1) is is in first, body, or last paragraph
        
        FUNC()
        # We can get 1 by checking the local id of each ac and its paragraph_id, if both
        # we just need to know hte nr of paragaraphs in each document,
        # then we can make conditions
        #               ac_para_id == 0 == FIRST
        #               ac_para_id == nr_para == LAST
        #               else: BODY

        # feature representation
        
        alt 1:

        one hot encodig where dimN == {0,1}, dim size = 4
        dim0 = 1 if item is first in sample 0 if not
        dim1 = 1 if item is in first paragraph if not
        dim2 = 1 if item is in body 0 if not
        dim3 = 1 if item is in last paragraph 0 if not


        alt 2: CURRENTLY USED!
        one hot encodig where dim0 == {0,1}
        dim0 = 1 if item is first in sample 0 if not

        and dim1 = {0,1,2}
        dim1 = 0 if item in first paragrap, 1 if in body, 2 if in last paragraph


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
        vec = np.zeros(2)
        
        doc_id = df["document_id"].max()

 
        # if unit is the first in the paragraph
        if df[f"paragraph_{self.prediction_level}_id"].max() == 0:
            vec[0]= 1
        

        # if unit is the first paragraph
        if df[f"document_paragraph_id"].max() == 0:
            pass
        

        elif df[f"document_paragraph_id"].max() == self.doc2paralen[doc_id]-1:
            vec[1] = 2

        # if unit is not in the first or last, its in the body
        else:
            vec[1] = 1
        
        return vec