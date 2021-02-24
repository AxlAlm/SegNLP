
#basics
import numpy as np
import string

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

#hotam
from hotam.nn.layers.char_emb import CHAR_EMB_LAYER
from hotam.nn.layers.lstm import LSTM_LAYER
from hotam.utils import zero_pad

# use a torch implementation of CRF
from torchcrf import CRF



class LSTM_CNN_CRF(nn.Module):

    """

    BiLSTM CNN CRF network
    
    ARGUMENT MINING PAPER
    paper1:
    https://arxiv.org/pdf/1704.06104.pdf

    ORIGINAL PAPER INTRODUCING NETWORK
    paper2:
    https://www.aclweb.org/anthology/P16-1101.pdf


    Network FLOW:

    1) get word embeddings

    2)  get character embeddings by using a CNN layer.
        character embeddings are then pooled to get a representation
        in the same dimension as words. I.e. from character embeddings
        we create a word embedding
    
    3) concatenate Word and Character-(word)-embeddings

    4) pass concatenated embeddings to bidirectional LSTM

    5) pass LSTM output to a linear layer to get logits for each class

    6) pass output to CRF

    7) voila: predictions

    
    STagT system corrections (SHOULD WE DO THIS OR NOT?):
        1) Invalid BIO structure, i.e., “I” follows “O”.

        2)  A predicted component is not homogeneous:for example, one token is predicted to link
            to the following argument component, whileanother token within the same component is
            predicted to link to the preceding argumentcomponent.

        3) A link goes ‘beyond’ the actual text, e.g.,
            when a premise is predicted to link to anothercomponent at ‘too large’ distance|d|.

    NOTE! 
    this network supports multitask learning by using seperate CRF layers per task.

    """ 

    def __init__(self, hyperparamaters:dict, task_dims:dict, feature_dims:dict):
        super().__init__()
        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        #self.DROPOUT = hyperparamaters["dropout"]
        self.HIDDEN_DIM = hyperparamaters["hidden_dim"]
        self.CHAR_DIM = hyperparamaters["char_dim"]
        self.KERNEL_SIZE = hyperparamaters["kernel_size"]
        self.NUM_LAYERS = hyperparamaters["num_layers"]
        self.BI_DIR = hyperparamaters["bidir"]

        self.WORD_EMB_DIM = feature_dims["word_embs"]

        self.lstm = LSTM_LAYER(  
                                input_size=self.WORD_EMB_DIM+self.CHAR_DIM,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                )
        
        #char_vocab = len(self.dataset.encoders["chars"])
        nr_chars = len(string.printable)+1
        self.char_cnn  = CHAR_EMB_LAYER(  
                                        vocab = nr_chars, 
                                        emb_size = self.CHAR_DIM,
                                        kernel_size = self.KERNEL_SIZE
                                        )

        #output layers. One for each task
        self.output_layers = nn.ModuleDict()

        #Crf layers, one for each task
        self.crf_layers = nn.ModuleDict()


        for task, output_dim in task_dims.items():
            self.output_layers[task] = nn.Linear(self.HIDDEN_DIM*(2 if self.BI_DIR else 1), output_dim)
            self.crf_layers[task] = CRF(num_tags=output_dim, batch_first=True)


    @classmethod
    def name(self):
        return "LSTM_CNN_CRF"


    def forward(self, batch):

        lengths = batch["lengths_tok"]
        mask = batch["token_mask"]

        #1
        word_embs = batch["word_embs"]

        #2
        char_embs = self.char_cnn(batch["chars"])

        #3
        cat_emb = torch.cat((word_embs, char_embs), dim=-1)

        #4 feed packed to lstm
        lstm_out, _ = self.lstm(cat_emb, lengths)

        tasks_preds = {}
        tasks_loss = {}
        tasks_probs = {}
        for task, output_layer in self.output_layers.items():

            target_tags = batch[task]

            #5
            dense_out = output_layer(lstm_out)
            
            #6
            crf = self.crf_layers[task]
            loss = -crf( 
                        emissions=dense_out,
                        tags=target_tags,
                        mask=mask,
                        reduction='mean'
                        )

            #7
            preds = crf.decode( 
                                emissions=dense_out, 
                                mask=mask
                                )

            tasks_preds[task] = torch.tensor(zero_pad(preds), dtype=torch.long)
            tasks_loss[task] = loss

        return {    
                    "loss":tasks_loss, 
                    "preds":tasks_preds,
                    "probs": {}
                }



