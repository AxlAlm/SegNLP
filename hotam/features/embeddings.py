#basics
from tqdm import tqdm
import os
from typing import List, Tuple, Union
import threading
import time

#torch
import torch
import numpy as np

#flair
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings, ELMoEmbeddings
from flair.data import Sentence
from flair.embeddings import Embeddings as FlairBaseEmbeddings

#hotam
from hotam import get_logger
from hotam.utils import ensure_numpy, timer
from hotam.features.base import FeatureModel, feature_memory
import hotam


logger = get_logger(__name__)

class Embeddings(FeatureModel):

    """
    class for extracting embeddings. Used Flair as it provides a ready made lib for extraction.

    Note that even though this model can extract embeddings from BERT and so on, this class
    is not trainable and cannot be use for fine tuning.

    """
    
    def __init__(self, emb_name:str, gpu:int=None):

        if isinstance(gpu, int):
            flair.device = torch.device(f'cuda:{gpu}') 

        self._name = emb_name.lower()
        self._context = "sentence" if emb_name in ["glove","flair","bert","elmo"] else False
        self._level = "word"
        self._input = "sentence"
        self._dtype = np.float32
        self.activate()


    def __get_embedder(self):
        """
        """


        if self._name == "bert":
            self.embedder = TransformerWordEmbeddings(
                                                    "bert-base-cased", 
                                                    layers="-1,-2,-3,-4", 
                                                    use_scalar_mix=False,
                                                    pooling_operation="mean"
                                                    )

        elif self._name == "flair":
            flair_forward_embedding = FlairEmbeddings('mix-forward')
            flair_backward_embedding = FlairEmbeddings('mix-backward')
            self.embedder = StackedEmbeddings(embeddings=[flair_forward_embedding, flair_backward_embedding])

        elif self._name == "elmo":
            self.embedder = ELMoEmbeddings()

        else:
            self.embedder = WordEmbeddings(self._name)


    def __get_word_embeddings(self, tokens):
        item_string = " ".join(tokens)
        flair_obj = Sentence(item_string, use_tokenizer=lambda x:x.split(" "))
        self.embedder.embed(flair_obj)
        embs = torch.stack([t.embedding for t in flair_obj])
        return embs


    # def __check_if_shutdown_ok(self):
    #     while SAVING_FEATURE != "done":
    #         time.sleep(30)
    #     self.deactivate()


    def deactivate(self):
        if not hasattr(self, "embedder"):
            raise Warning("Model is not active, so cannot deactivate")
        logger.info(f"{self._name}-emb model will be deactivated")
        del self.embedder


    def activate(self):
        logger.info(f"{self._name}-emb model is not active, will load model")
        self.__get_embedder()
        self._feature_dim = self.embedder.embedding_length

        #start a new thread which just checks every 30 sec if features are 
        # all saved, if they are they will deactivate the model so it doesnt 
        # take up memory
        # if hotam.preprocessing.settings["SAVE_FETURES"]:
        #     x = threading.Thread(target=self.__check_if_shutdown_ok)
        #     x.start()


    @feature_memory
    def extract(self, df) -> np.ndarray:
        tokens = df["text"].to_numpy()

        # self.__time_last_used = time.time()
        if not hasattr(self, "embedder"):
            self.activate()


        return ensure_numpy(self.__get_word_embeddings(tokens))


