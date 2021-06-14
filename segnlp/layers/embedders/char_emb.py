

#basics
import string

#pytroch
import torch.nn as nn
from torch import Tensor


class CharEmb(nn.Module):

    def __init__(   
                self,
                n_filters:int,
                emb_size:int,
                kernel_size:int,
                dropout:float=0.0,
                ):
        super().__init__()
        self.char_emb_layer = nn.Embedding( 
                                            num_embeddings=len(string.printable)+1,
                                            embedding_dim=emb_size, 
                                            padding_idx=0
                                            )
        self.char_cnn = nn.Conv1d(  
                                    in_channels=emb_size, 
                                    out_channels=n_filters, 
                                    kernel_size=kernel_size, 
                                    )
        self.dropout = nn.Dropout(dropout)
        self.output_size = n_filters
                                    

    def forward(self, char_encs:Tensor):
        batch_size, seq_length, char_length = char_encs.shape

        char_embs = self.char_emb_layer(char_encs)
        char_embs = self.dropout(char_embs)

        # char emb is a 4D vector of Batch size, seq length, character length and embedding size
        # we want to Conv over character embeddings per word, per sequence, so we can move the dimension 1 step down
        # by batch_size x seq_length. Batchsize is now each words and we cant run our CONV over the character embeddings
        # we are in practive treaing each word as its own batch
        char_embs = char_embs.view(-1, char_embs.shape[-1], char_length) #.transpose(1, 2)
        cnn_out = self.char_cnn(char_embs)

        # we can now pool the embeddings for each sequence of character embeddings and create word embeddings
        values, indices = cnn_out.max(dim=2)

        # as we have a stack of word embeddings we can now format it back in a view 
        # that has batchsize, seq_length and n_filters, 
        word_embs = values.view(batch_size, seq_length, values.shape[-1])

        return word_embs