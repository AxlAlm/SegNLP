

# basics
from segnlp.data import batch
from typing import List, Sequence
import h5py
import json
import numpy as np
import os
from functools import wraps
import pwd


#flair
from flair.embeddings import WordEmbeddings
from flair.embeddings import FlairEmbeddings as FlairFlairEmbeddings
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.embeddings import Embeddings as FlairBaseEmbeddings
from flair.data import Sentence
import flair 


# pytorch
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# segnlp
from segnlp import utils 


# TEMPORARY
# NOTE! tie to set device
flair.device = torch.device('cpu') 


user_dir = pwd.getpwuid(os.getuid()).pw_dir


class FlairEmbeddings(nn.Module):

    def __init__(self, embs:str) -> None:
        super().__init__()
        self._list_embs = embs.split("+")
        self.embedder = self.__create_embedder()
        self.output_size = self.embedder.embedding_length
        self.__init__h5py_storage()

        #freeze all of the paramaters
        for name, param in self.named_parameters():
            param.requires_grad = False


    def __init__h5py_storage(self):
        self._storage_dir = f"{user_dir}/.segnlp/flair_embeddings/{'_'.join(self._list_embs)}"
        os.makedirs(self._storage_dir, exist_ok = True)

        self._h5py_fp = os.path.join(self._storage_dir, "data.h5")
        self._sent2idx_fp = os.path.join(self._storage_dir, "sent2idx.txt")

        if os.path.exists(self._h5py_fp):
            self._h5py_file = h5py.File(self._h5py_fp, "a")
            self._storage = self._h5py_file["embs"]

            self._sent2idx = {sent:int(i) for sent,i in utils.read_file(self._sent2idx_fp, line_fn=lambda x: x.split("\t") )}

            self.i = len(self._sent2idx_fp)
        else:
            self._h5py_file = h5py.File(self._h5py_fp, "a")
            self._storage = self._h5py_file.create_dataset(
                                            "embs",
                                            data=np.zeros((0,0, self.output_size)), 
                                            maxshape=(None,None, None),
                                            dtype=np.float,
                                            fillvalue = 0
                                            )

            self._sent2idx = {}
            self.i = 0
        

    def _h5py_cache(fn):

        @wraps(fn)
        def wrapper(self, sentence):
            
            if sentence in self._sent2idx:
                return torch.tensor(self._storage[self._sent2idx[sentence]])
            else:
                # get the output
                output = utils.ensure_numpy(fn(self, sentence))

                # get the new shape and update the h5py file
                batch_size, currnet_n_toks, *_ = self._storage.shape
                n_toks, *_ = output.shape
                new_shape = (batch_size+1, max(n_toks, currnet_n_toks), self.output_size)
                self._storage.resize(new_shape)   

                # cache output
                self._storage[self.i, :n_toks] = output
                utils.write_data(self._sent2idx_fp, f"{sentence}\t{self.i}\n", mode = "a")
                self._sent2idx[sentence] = self.i


                self.i += 1

                return torch.tensor(output)
        
        return wrapper

    
    def __create_embedder(self) -> FlairBaseEmbeddings:

        # create 
        embs = []
        if "flair" in self._list_embs:
            flair_embedding_forward = FlairFlairEmbeddings('news-forward')
            flair_embedding_backward = FlairFlairEmbeddings('news-backward')
            flair_embedding = StackedEmbeddings(embeddings=[flair_embedding_forward, flair_embedding_backward])
            embs.append(flair_embedding)

        
        if "bert" in self._list_embs:
            bert_embeddings = TransformerWordEmbeddings('bert-base-uncased', layers='all', layer_mean=True)
            embs.append(bert_embeddings)


        if "glove" in self._list_embs:
            glove_embs = WordEmbeddings("glove")
            embs.append(glove_embs)

        
        if len(embs) > 1:
            embedder = StackedEmbeddings(embeddings=embs)
        else:
            embedder = embs[0]

        return embedder


    @_h5py_cache
    def _process_sentence(self, sentence:str) -> Tensor:
        flair_obj = Sentence(sentence, use_tokenizer=lambda x:x.split(" ")) 
        self.embedder.embed(flair_obj)
        return torch.stack([t.embedding for t in flair_obj])


    def forward(self, input: List[List[str]]) -> Tensor:
        try:
            return torch.stack([self._process_sentence(sentence) for sentence in input], dim=0).type(torch.float)
        except RuntimeError:
            return pad_sequence(
                            [self._process_sentence(sentence) for sentence in input], 
                            batch_first=True,
                            padding_value=0.0
                            ).type(torch.float)

    




    # def preprocess_dataset(self, list_sample:List[str]):

    #     sample2idx = {}
    #     max_tok_len = max(len(s.split(" ")) for s in list_sample)
    #     embedding_matrix = np.zeros((len(list_sample), max_tok_len, self.embedder.embedding_length))


    #     for i, sample in enumerate(list_sample):


    #         if sample in sample2idx:
    #             continue

    #         sample2idx[sample] = i

    #         flair_obj = Sentence(sample, use_tokenizer=lambda x:x.split(" "))
    #         self.embedder.embed(flair_obj)
    #         embs = np.stack([t.embedding for t in flair_obj])
    #         #print(embs)
    #         embedding_matrix[i] = embs

        
    #     with h5py.File("/tmp/flair_embs.hdf5", "w") as f:
    #         f.create_dataset(
    #                         "embs", 
    #                         data = embedding_matrix, 
    #                         dtype=embedding_matrix.dtype
    #                         )


    #     with open("/tmp/sample2idx.json", "w") as f:
    #         json.dump(sample2idx, f)
