

#basics
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#segnlp
from segnlp.nn.layers.rep_layers import LSTM
from segnlp.nn.layers.link_layers import PairingLayer
import segnlp.utils as u


class LSTM_DIST(nn.Module):

    """

    Implementation of an LSTM-Minus-based span representation network
    for Argument Structre Parsing / Argument Mining

    paper:
    https://www.aclweb.org/anthology/P19-1464/

    Original code:
    https://github.com/kuribayashi4/span_based_argumentation_parser/tree/614343b18e7d98293a2b020f9ab05b86355e18df


    More on LSTM-Minus
    https://www.aclweb.org/anthology/P16-1218/



    Model Overview:
    
    1) pass word embeddigns to an LSTM, get hidden reps H.

    2) given H create LSTM-minus representations for AM anc AC

    3) pass the AM and AC minus representations to seperate BiLSTMs

    4) Concatenate AM and AC output from 3) with document embeddings (BOW and document positions)

    5) Classification 
            
            AC types: output of 4 to a linear layer

            Stance/link Type: output of 4 to a linear layer

            Link/relation:  see layers/PairingLayer

    """

    def __init__(self, hyperparamaters:dict, task_dims:dict, feature_dims:dict, inference:bool):
        super().__init__()
        self.inference = inference
        self.BATCH_SIZE = hyperparamaters["batch_size"]
        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.HIDDEN_DIM = hyperparamaters["hidden_dim"]
        self.NUM_LAYERS = hyperparamaters["num_layers"]
        self.BI_DIR = hyperparamaters["bidir"]

        self.loss_weight = hyperparamaters["loss_weight"]
        
        self.WORD_FEATURE_DIM = feature_dims["word_embs"]
        self.DOC_FEATURE_DIM = feature_dims["doc_embs"]


        self.word_lstm = LSTM(  
                                input_size = self.WORD_FEATURE_DIM,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                dropout= hyperparamaters["input_dropout"],
                                w_init="orthogonal"

                                )


        self.am_lstm = LSTM(  
                                input_size = self.HIDDEN_DIM*4,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                dropout=hyperparamaters["lstm_dropout"],
                                w_init="orthogonal"
                                )

        self.ac_lstm = LSTM(  
                                input_size = self.HIDDEN_DIM*4,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                dropout=hyperparamaters["lstm_dropout"],
                                w_init="orthogonal"

                                )

        am_ac_bow_size = (self.HIDDEN_DIM*(2 if self.BI_DIR else 1) * 2) + self.DOC_FEATURE_DIM
        self.am_ac_lstm = LSTM(  
                                input_size = am_ac_bow_size,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                dropout=hyperparamaters["lstm_dropout"],
                                w_init="orthogonal"

                                )

        self.last_lstm = LSTM(  
                                input_size = self.HIDDEN_DIM*2,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                dropout=hyperparamaters["lstm_dropout"],
                                w_init="orthogonal"

                                )

        self.output_dropout = nn.Dropout(hyperparamaters["output_dropout"])
        
        self.link_label_clf = nn.Linear(self.HIDDEN_DIM*2, task_dims["link_label"])
        torch.nn.init.uniform_(self.link_label_clf.weight.data,  a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.link_label_clf.bias.data,  a=-0.05, b=0.05)

        self.label_clf = nn.Linear(self.HIDDEN_DIM*2, task_dims["label"])
        torch.nn.init.uniform_(self.label_clf.weight.data,  a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.link_label_clf.bias.data,  a=-0.05, b=0.05)


        self.link_clf = PairingLayer(
                                    input_dim=self.HIDDEN_DIM*2, 
                                    max_units=task_dims["link"],
                                    dropout=hyperparamaters["output_dropout"]
                                    )

        self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)


    @classmethod
    def name(self):
        return "LSTM_DIST"


    def forward(self, batch, output):
        
        self.device = batch.device
        word_embs = batch["token"]["word_embs"]

        # Wi:j 
        W = batch["unit"]["doc_embs"]

        # batch is sorted by length of prediction level which is Argument Components
        # so we need to sort the word embeddings for the sample, pass to lstm then return to 
        # original order
        sorted_lengths_tok, sorted_indices = torch.sort(batch["token"]["lengths"], descending=True)
        _ , original_indices = torch.sort(sorted_indices, descending=False)

        # 1
        # input (Batch_dize, nr_tokens, word_emb_dim)
        # output (Batch_dize, nr_tokens, word_emb_dim)
        lstm_out, _ = self.word_lstm(batch["token"]["word_embs"][sorted_indices], sorted_lengths_tok)
        lstm_out = lstm_out[original_indices]

        # 2
        # create span representation for Argument Components and Argumentative Markers
        am_minus_embs = self.__minus_span(lstm_out, batch["am"]["span_idxs"])
        ac_minus_embs = self.__minus_span(lstm_out, batch["unit"]["span_idxs"])

        # 3
        # pass each of the spans to a seperate BiLSTM. 
        # NOTE! as Argumentative Markers are not allways present the length will sometimes be 0, 
        # which will cause an error when useing pack_padded_sequence etc.
        # to fix this all AM's that are non existing are set to a default lenght of 1.
        # this will not really matter as these representations will be 0s anyway.
        #
        # we also need to sort AMS so they can be passed to the lstm
        sorted_am_lengths, sorted_am_indices = torch.sort(batch["am"]["lengths"], descending=True)
        _ , original_am_indices = torch.sort(sorted_indices, descending=False)
        am_lstm_out, _ = self.am_lstm(am_minus_embs[sorted_indices], sorted_am_lengths)
        am_lstm_out = am_lstm_out[original_am_indices]

        ac_lstm_out, _ = self.ac_lstm(ac_minus_embs, batch["unit"]["lengths"])

        # 4
        # concatenate the output from Argument Component BiLSTM and Argument Marker BiLSTM with BOW embeddigns W
        cat_emb = torch.cat((am_lstm_out, ac_lstm_out, W), dim=-1)
        adu_emb, _= self.am_ac_lstm(cat_emb, batch["unit"]["lengths"])

        #5
        adu_emb = self.output_dropout(adu_emb)

        # Classification of AC and link labels is pretty straight forward
        link_label_out = self.link_label_clf(adu_emb)
        label_out = self.label_clf(adu_emb)

        
        adu_emb, _ = self.last_lstm(adu_emb, batch["unit"]["lengths"])
        adu_emb = self.output_dropout(adu_emb)

        link_out = self.link_clf(adu_emb, unit_mask=batch["unit"]["mask"])

        link_preds = torch.argmax(link_out, dim=-1)
        link_label_preds = torch.argmax(link_label_out, dim=-1)
        label_preds = torch.argmax(label_out, dim=-1)

     
        if not self.inference:   
            link_loss = self.loss(torch.flatten(link_out, end_dim=-2), batch["unit"]["link"].view(-1))
            link_label_loss = self.loss(torch.flatten(link_label_out, end_dim=-2), batch["unit"]["link_label"].view(-1))
            label_loss = self.loss(torch.flatten(label_out, end_dim=-2), batch["unit"]["label"].view(-1))

            ## this is the reported loss aggregation in the paper, but...
            #total_loss = -((self.loss_weight * link_loss) + (self.loss_weight * label_loss) + ( (1 - self.loss_weight- self.loss_weight) * link_label_loss))
            ## in the code the loss is different
            ## https://github.com/kuribayashi4/span_based_argumentation_parser/blob/614343b18e7d98293a2b020f9ab05b86355e18df/src/classifier/parsing_loss.py#L88-L91
            total_loss = ((1 - self.loss_weight - self.loss_weight) * link_loss) - (self.loss_weight * label_loss) + (self.loss_weight * link_label_loss)



            output.add_loss(task="total",       data=total_loss)
            output.add_loss(task="link",        data=link_loss)
            output.add_loss(task="link_label",  data=link_label_loss)
            output.add_loss(task="label",       data=label_loss)


        #print(link_preds[3] ,batch["unit"]["link"][3])
        output.add_preds(task="label",          level="unit", data=label_preds)
        output.add_preds(task="link",           level="unit", data=link_preds)
        output.add_preds(task="link_label",     level="unit", data=link_label_preds)

        # output.add_preds(task="label",          level="unit", data=batch["unit"]["label"])
        # output.add_preds(task="link",           level="unit", data=batch["unit"]["link"])
        # output.add_preds(task="link_label",     level="unit", data=batch["unit"]["link_label"])

        return output