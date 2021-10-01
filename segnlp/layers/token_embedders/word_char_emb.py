

#basics
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

class WordCharEmb(nn.Module):

    """
    Creating word embeddings from aggregation of character embeddings
    
    """

    def __init__(   
                self,
                num_embeddings:int,
                kernel_size:int,
                n_filters:int,
                ):
        super().__init__()
        self.embs = Embs(
                        vocab = "Char",
                        num_embeddings = num_embeddings
                        )
        self.char_cnn = nn.Conv1d(  
                                    in_channels=self.embs.output_size, 
                                    out_channels=n_filters, 
                                    kernel_size=kernel_size, 
                                    )
        self.output_size = n_filters
                                    

    def forward(self, 
                input:Sequence, 
                lengths:Sequence,
                ):
    
        embs = self.embs(input)
        
        #encode tokens and padd to longest word
        char_encs_flat = pad_sequence(self.vocab[input], batch_first = True)

        # split all words into samples and padd to longest sample
        char_encs = pad_sequence(
                                torch.split(
                                            char_encs_flat, 
                                            utils.ensure_list(lengths)
                                            ), 
                                batch_first = True
                                )

        batch_size, seq_length, char_length = char_encs.shape

        char_embs = self.char_emb_layer(char_encs)

        # char emb is a 4D vector of Batch size, seq length, character length and embedding size
        # we want to Conv over character embeddings per word, per sequence, so we can move the dimension 1 step down
        # by batch_size x seq_length. Batchsize is now each words and we cant run our CONV over the character embeddings
        # we are in practive treaing each word as its own batch
        char_embs = char_embs.view(-1, char_embs.shape[-1], char_length) #.transpose(1, 2)
        cnn_out = self.char_cnn(char_embs)

        # we can now pool the embeddings for each sequence of character embeddings and create word embeddings
        values, _ = cnn_out.max(dim=2)

        # as we have a stack of word embeddings we can now format it back in a view 
        # that has batchsize, seq_length and n_filters, 
        word_embs = values.view(batch_size, seq_length, values.shape[-1])

        return word_embs