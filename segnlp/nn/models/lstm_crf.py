
#basics
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#segnlp
from segnlp.nn.layers import LSTM_LAYER
from segnlp.utils import zero_pad
import segnlp.utils as u

# use a torch implementation of CRF
from torchcrf import CRF


'''
###MODEL DESCRIPTION### 
#######################

'''

class LSTM_CRF(nn.Module):

    """
    https://www.aclweb.org/anthology/W19-4501

    """

    def __init__(self, hyperparamaters:dict, task_dims:dict, feature_dims:dict, inference:bool):
        super().__init__()
        self.inference = inference
        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.HIDDEN_DIM = hyperparamaters["hidden_dim"]
        self.NUM_LAYERS = hyperparamaters["num_layers"]
        self.BI_DIR = hyperparamaters["bidir"]
        self.FINETUNE_EMBS = hyperparamaters["fine_tune_embs"]
        self.WORD_EMB_DIM = feature_dims["word_embs"]


        #dropout
        self.use_dropout = False
        if "dropout" in hyperparamaters:
            self.dropout = nn.Dropout(hyperparamaters["dropout"])
            self.use_dropout = True

        # model is a reimplementation of Flairs.SequenceTagger (LINK).
        # Model is implemented to replicate the scores from ().
        # Why finetuning of embeddings is done like this is, i assume, because
        # the finetuning of multiple models, ELMO BERT FLAIR GLOVE, etc... in cases of concatenated 
        # embeddigns would be very difficult and not very usefull. Hence a small remapping at least 
        # produces some sort of finetuning of the combination of embeddings used.
        #
        # https://github.com/flairNLP/flair/issues/1433
        # https://github.com/flairNLP/flair/commit/0056b2613ee9c169cb9c23e5e84fbcca180dde77#r31462803
        if self.FINETUNE_EMBS:
            self.emb2emb = torch.nn.Linear(self.WORD_EMB_DIM, self.WORD_EMB_DIM)

        self.lstm = LSTM_LAYER(  
                            input_size = self.WORD_EMB_DIM,
                            hidden_size=self.HIDDEN_DIM,
                            num_layers= self.NUM_LAYERS,
                            bidirectional=self.BI_DIR,
                            )

        #output layers. One for each task
        self.output_layers = nn.ModuleDict()

        #Crf layers, one for each task
        self.crf_layers = nn.ModuleDict()

        for task, output_dim in task_dims.items():

            self.output_layers[task] = nn.Linear(self.HIDDEN_DIM*(2 if self.BI_DIR else 1), 
                                                output_dim)
            self.crf_layers[task] = CRF(    
                                            num_tags=output_dim,
                                            batch_first=True
                                        )


    @classmethod
    def name(self):
        return "LSTM_CRF"


    def forward(self, batch, output):

        lengths = batch["token"]["lengths"]
        mask = batch["token"]["mask"]
        word_embs = batch["token"]["word_embs"]

        if self.use_dropout:
            word_embs = self.dropout(word_embs)

        if self.FINETUNE_EMBS:
            word_embs = self.emb2emb(word_embs)
        
        lstm_out, _ = self.lstm(word_embs, lengths)

        for task, output_layer in self.output_layers.items():

            dense_out = output_layer(lstm_out)

            crf = self.crf_layers[task]
            
            if not self.inference:
                # crf doesnt work if padding is not a possible class, so we put padding as 0 which
                # will be default to "O" in BIO or "None" (see label encoders)                
                batch.change_pad_value(level="token", task=task, new_value=0)
                
                target_tags = batch["token"][task]
                loss = -crf(    
                                emissions=dense_out, #score for each tag, (batch_size, seq_length, num_tags) as we have batch first
                                tags=target_tags,
                                mask=mask,
                                reduction='mean'
                                )
                output.add_loss(task=task,   data=loss)


            #returns preds with no padding (padding values removed)
            preds = crf.decode( 
                                emissions=dense_out, 
                                mask=mask
                                )
            
            output.add_preds(task=task, level="token", data=torch.tensor(zero_pad(preds), dtype=torch.long))
        
        return output