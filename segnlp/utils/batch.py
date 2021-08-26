
#basics
import numpy as np


# segnlp
from segnlp import utils


class Level:

    @cache
    def any_key(self, key):
        flat_values = self.df.loc[:, key].to_numpy()
        splits = np.split(flat_values, self.lengths())
        utils.pad(splits, pad_value = -1 if key in tasks else 0)


class TokenLevel(Level):

    @cache
    def lengths(self):
        self.df.groupby(level=0, sort = False).size()

    @cache
    def mask(self):
        return utils.create_mask(self.lengths(), as_bool = True) 


class SegLevel(Level):
    
    @cache
    def lengths(self,):
        return self.df.groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()


    def lengths_tok():
        return self.df.groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()
        self.df.groupby(level=0, sort = False).size()
        

    def span_idxs():
        return self.df.groupby(level=0, sort=False)["seg_id"].nunique().to_numpy()



class SpanLevel(Level):
        pass



class Batch(dict):


    def __init__(self, 
                df: pd.Dataframe, 
                word_embs: np.ndarray = None,
                seg_embs: np.ndarray = None,
                ):
        
        self._df = df
        # self._word_embs = word_embs
        # self._seg_embs = seg_embs

        self["token"] = TokenLevel(self._df, word_embs = word_embs)
        self["seg"] = SegLevel(self._df, seg_embs = seg_embs)
        self["span"] = SpanLevel(self._df)

    # @property
    # def df(self):
    #     return self.df

    # @property
    # def word_embs(self):
    #     return self._word_embs

    # @property
    # def seg_embs(self):
    #     return self._seg_embs
