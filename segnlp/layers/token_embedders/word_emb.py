

#basics
import string
from typing import Union
from tqdm.auto import tqdm

#pytroch
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

#gensim
from gensim.models import KeyedVectors

#segnlp
from segnlp.resources.vocab import Vocab
from segnlp import resources
from segnlp import utils


class WordEmb(nn.Module):

    """
    Layer for embedding words. 

    For loading pretrained embeddings a vocab and a model.bin file needs to be provided. 
    A matrix of embeddings will be created for the vocab which will initialize the embedding layer

    NOTE! make sure the vocab is the same as the vocab used to encode the words

    
    You can find some models here:

    http://vectors.nlpl.eu/repository/

    https://fasttext.cc/docs/en/pretrained-vectors.html
    
    """

    def __init__(   
                self,
                vocab: Vocab,
                path_to_pretrained: str = None,
                num_embeddings: int = None,
                embedding_dim: int = None,
                freeze: bool = None,
                padding_idx: int = 0, 
                max_norm: float = None, 
                norm_type: float = 2.0, 
                scale_grad_by_freq:bool = False, 
                sparse = False,
                ):
        super().__init__()
        self.vocab = getattr(resources.vocab, vocab)() if isinstance(vocab, str) else vocab

        if path_to_pretrained is not None:
            # we create embedding matrix for the given vocab. in the order given
            vocab_matrix = self.__create_vocab_matrix(vocab, path_to_pretrained)

            self.embs = nn.Embedding.from_pretrained(
                                                        vocab_matrix, 
                                                        freeze=freeze
                                                        )
        else:
            self.embs = nn.Embedding( 
                                    num_embeddings = num_embeddings, 
                                    embedding_dim = embedding_dim, 
                                    padding_idx = padding_idx, 
                                    max_norm = max_norm, 
                                    norm_type = norm_type, 
                                    scale_grad_by_freq = scale_grad_by_freq, 
                                    sparse = sparse, 
                                    )
        
        self.output_size = self.embs.embedding_dim


    def __create_vocab_matrix(self, vocab:list, path_to_pretrained:str):

        model = KeyedVectors.load_word2vec_format(path_to_pretrained, binary=True if path_to_pretrained.endswith(".bin") else False)

        vocab_matrix = torch.zeros((len(vocab), model.vector_size))

        for i, word in vocab.vocab.items():

            if word not in model:
                vocab_matrix[i] = torch.rand(model.vector_size, dtype=torch.float)
                continue
            
            vocab_matrix[i] = torch.tensor(model[word], dtype=torch.float)

    
        return vocab_matrix
            

    def forward(self, input:list, lengths: list):
        
        # encode tokens, assumes input is a list of all tokens in the batch
        token_id_flat = self.vocab[input]

        # formats the encoded tokens to (Batch size, max token length)
        token_ids = pad_sequence(
                                torch.split(
                                            token_id_flat, 
                                            utils.ensure_list(lengths)
                                            ), 
                                        batch_first = True
                                )

        # gets embeddings
        embs = self.embs(token_ids)

        return embs