

#basics
from segnlp.layers import dropouts
import string
from typing import Sequence

#pytroch
import torch.nn as nn
from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence

# segnlp
from .embs import Embs
from segnlp import utils

class CharWordEmb(nn.Module):

    """
    Creating word embeddings from aggregation of character embeddings
    
    """

    def __init__(   
                self,
                embedding_dim:int,
                kernel_size:int,
                n_filters:int,
                dropout: float = 0.0,
                ):
        super().__init__()
        self.embs = Embs(
                        vocab = "Char",
                        embedding_dim = embedding_dim
                        )
        self.dropout = nn.Dropout(dropout)
        self.cnn = nn.Conv1d(  
                                    in_channels=self.embs.output_size, 
                                    out_channels=n_filters, 
                                    kernel_size=kernel_size, 
                                    )
        self.output_size = n_filters
                                    

    def forward(self, 
                input:Sequence, 
                lengths:Sequence,
                ):
        
        # first we embed each token in the back
        char_embs = self.embs(input)
    
        # then we split the words into sample
        char_embs = pad_sequence(
                                torch.split(
                                            char_embs, 
                                            utils.ensure_list(lengths)
                                            ), 
                                batch_first = True
                                )

        batch_size, max_tokens, max_char, emb_dim = char_embs.shape

        #apply dropout
        char_embs = self.dropout(char_embs)

        # char emb is a 4D vector of Batch size, seq length, character length and embedding size
        # we want to Conv over character embeddings per word, per sequence, we can do this by
        # changing dimension to (batch_size x max_tokens, emb_dim, max_char)
        # NOTE! input to a conv1d is suppose to be (batch, in_channels/embedding dim, length of sequence)
        # 
        # Batchsize is now each words and we cant run our CONV over the character embeddings
        # we are in practice treating each word as its own batch
        #
        # example 
        #   >> charater embeddings -> [3716, 17, 30] # 30 == embedding dim
        #   >> reshape -> [4470, 30, 17]
        #   >> cnn output - >[4470, 50, 15] # 50 = n_filters
        #   >> cnn max pool -> [4470, 50]
        #   >> max pool reshape -> [batch_size, max_tok, n_filters]
        char_embs = char_embs.view(-1, emb_dim, max_char) #.transpose(1, 2)

        #pass to cnn
        cnn_out = self.cnn(char_embs)

        # we can now pool the embeddings for each sequence of character embeddings and create word embeddings
        values, _ = cnn_out.max(dim=2)

        # as we have a stack of word embeddings we can now format it back in a view 
        # that has batchsize, seq_length and n_filters, 
        word_embs = values.view(batch_size, max_tokens, self.output_size)

        return word_embs