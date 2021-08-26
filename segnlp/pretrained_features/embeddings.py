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
from flair.embeddings import WordEmbeddings
from flair.embeddings import FlairEmbeddings as FlairFlairEmbeddings
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import ELMoEmbeddings as FLairELMoEmbeddings
from flair.embeddings import StackedEmbeddings

from flair.data import Sentence
from flair.embeddings import Embeddings as FlairBaseEmbeddings

#segnlp
from segnlp import get_logger
from segnlp.utils import ensure_numpy, timer
from segnlp.pretrained_features.base import FeatureModel
import segnlp

logger = get_logger(__name__)


class FlairEmbWrapper(FeatureModel):

    """
    class for extracting embeddings. Used Flair as it provides a ready made lib for extraction.

    Note that even though this model can extract embeddings from BERT and so on, this class
    is not trainable and cannot be use for fine tuning.

    """
    
    def __init__(self, flair_embedding, emb_name:str, gpu:int=None, group:str="word_embs"):
            
        if isinstance(gpu, int):
            flair.device = torch.device(f'cuda:{gpu}')
        else:
            flair.device = torch.device(f'cpu')

        self._name = emb_name.lower()
        self._context = "sentence" if emb_name in ["glove","flair","bert","elmo"] else False
        self._level = "word"
        self._dtype = np.float32
        self._group = self._name if group is None else group
        self.embedder = flair_embedding
        self._feature_dim = self.embedder.embedding_length


    def extract(self, df) -> np.ndarray:
        tokens = df["text"].to_numpy()
        item_string = " ".join(tokens)
        flair_obj = Sentence(item_string, use_tokenizer=lambda x:x.split(" "))
        self.embedder.embed(flair_obj)
        embs = torch.stack([t.embedding for t in flair_obj])
        return ensure_numpy(embs)




class FlairEmbeddings(FlairEmbWrapper):

    def __init__(self, gpu:int=None, group:str="word_embs"):
        flair_embedding_forward = FlairFlairEmbeddings('news-forward')
        flair_embedding_backward = FlairFlairEmbeddings('news-backward')
        flair_embedding = StackedEmbeddings(embeddings=[flair_embedding_forward, flair_embedding_backward])
        super().__init__(flair_embedding, emb_name="flair", gpu=gpu, group=group)



class BertEmbeddings(FlairEmbWrapper):

    def __init__(self, gpu:int=None, group:str="word_embs"):
        flair_embedding = TransformerWordEmbeddings('bert-base-uncased', layers='all', layer_mean=True)
        super().__init__(flair_embedding, emb_name="bert", gpu=gpu, group=group)



class GloveEmbeddings(FlairEmbWrapper):

    def __init__(self, gpu:int=None, group:str="word_embs"):
        flair_embedding = WordEmbeddings("glove")
        super().__init__(flair_embedding, emb_name="glove", gpu=gpu, group=group)



class ELMoEmbeddings(FlairEmbWrapper):

    def __init__(self, gpu:int=None, group:str="word_embs"):
        flair_embedding = FLairELMoEmbeddings(model="original", embedding_mode="average")
        super().__init__(flair_embedding, emb_name="elmo", gpu=gpu, group=group)





