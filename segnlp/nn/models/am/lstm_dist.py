

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
        self.ALPHA = hyperparamaters["alpha"]
        self.BETA = hyperparamaters["beta"]
        
        self.WORD_FEATURE_DIM = feature_dims["word_embs"]
        self.DOC_FEATURE_DIM = feature_dims["doc_embs"]

        # if this is true we will make a distinction between ams and acs
        #self.DISTICTION = hyperparamaters["distinction"]

        self.word_lstm = LSTM(  
                                input_size = self.WORD_FEATURE_DIM,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                )


        self.am_lstm = LSTM(  
                                input_size = self.HIDDEN_DIM*3,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                )

        self.ac_lstm = LSTM(  
                                input_size = self.HIDDEN_DIM*3,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                )

        # input to adu_lstm is the am + ac + doc features 
        adu_shape = ((self.HIDDEN_DIM*(2 if self.BI_DIR else 1)) * 2 ) + self.DOC_FEATURE_DIM

        # self.adu_lstm = LSTM_LAYER(  
        #                             input_size = adu_shape,
        #                             hidden_size=self.HIDDEN_DIM,
        #                             num_layers= self.NUM_LAYERS,
        #                             bidirectional=self.BI_DIR,
        #                             )

        input_size = (self.HIDDEN_DIM*(2 if self.BI_DIR else 1) * 2) + self.DOC_FEATURE_DIM
        self.link_label_clf = nn.Linear(input_size, task_dims["link_label"])
        self.label_clf = nn.Linear(input_size, task_dims["label"])
        self.link_clf = PairingLayer(input_dim=input_size, max_units=task_dims["link"])

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)


    @classmethod
    def name(self):
        return "LSTM_DIST"


    def __minus_span(self, bidir_lstm_output, spans):

        """
        based on paper:
        https://www.aclweb.org/anthology/P19-1464.pdf
        (above paper refers to: https://www.aclweb.org/anthology/P16-1218.pdf)

        Minus based representations are away to represent segments given a representations
        of the segment as a set of e.g. tokens. We subtract LSTM hidden vectors to create a 
        vector representation of a segment.

        NOTE! in https://www.aclweb.org/anthology/P19-1464.pdf:
        "Each ADU span is denoted as (i, j), where i and j are word indices (1 ≤ i ≤ j ≤ T)."
        but for to make the spans more pythonic we define them as  (0 ≤ i ≤ j < T)

        (NOTE!  φ(wi:j), whihc is included in the forumla i the paper,  is added later because of Distiction between AM and AC)

        minus span reps are following:
            h(i,j) = [
                        →hj − →hi-1;
                        ←hi − ←hj+1; 
                        →hi-1;← hj+1    
                        ]

        NOTE! we assume that ←h is aligned with →h, i.e.  ←h[i] == →h[i] (the same word)
        
        So, if we have the following hidden outputs; 

            fh (forward_hidden)   = [word0, word1, word2, word3, word4]
            bh (backwards hidden) = [word0, word1, word2, word3, word4] 
        
        h(2,4) = [
                    word4 - word1;
                    word2 - word5;
                    word1 - word5
                    ]

        NOTE! word5 will be all zeros

        So, essentially we take previous and next segments from and subtract them from the current segment to create 
        a representation that include information about previous and next segments :D

        """

        batch_size, nr_seq, bidir_hidden_dim = bidir_lstm_output.shape
        hidden_dim = int(bidir_hidden_dim/2)
        feature_dim = hidden_dim*3

        forward = bidir_lstm_output[:,:,:hidden_dim]
        backward = bidir_lstm_output[:,:,hidden_dim:]

        minus_reps = torch.zeros((batch_size, nr_seq, feature_dim), device=self.device)
        
        for idx in range(batch_size):
            for sidx,(i,j) in enumerate(spans[idx]):


                if i==0 and j == 0:
                    continue

                if i-1 == -1:
                    f_pre = torch.zeros(hidden_dim, device=self.device)
                else:
                    f_pre = forward[idx][i-1]


                if j+1 > backward.shape[1]:
                    b_post = torch.zeros(hidden_dim,  device=self.device)
                else:
                    b_post = backward[idx][j+1]

                f_end = forward[idx][j]
                b_start = backward[idx][i]

                minus_reps[idx][sidx] = torch.cat((
                                                    f_end - f_pre,
                                                    b_start - b_post,
                                                    f_pre - b_post
                                                    ))

        return minus_reps


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
        # to fix this all AM's that are non existing are set to a defualt lenght of 1.
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
        contex_emb = torch.cat((am_lstm_out, ac_lstm_out, W), dim=-1)


        # 5
        # Classification of AC and link labels is pretty straight forward
        link_label_out = self.link_label_clf(contex_emb)
        label_out = self.label_clf(contex_emb)
        link_out = self.link_clf(contex_emb, unit_mask=batch["unit"]["mask"])

        #link_label_probs = F.softmax(link_label_out, dim=-1)
        #label_probs = F.softmax(label_out, dim=-1)

        link_preds = torch.argmax(link_out, dim=-1)
        link_label_preds = torch.argmax(link_label_out, dim=-1)
        label_preds = torch.argmax(label_out, dim=-1)


        
        if not self.inference:   
            link_loss = self.loss(torch.flatten(link_out, end_dim=-2), batch["unit"]["link"].view(-1))
            link_label_loss = self.loss(torch.flatten(link_label_out, end_dim=-2), batch["unit"]["link_label"].view(-1))
            label_loss = self.loss(torch.flatten(label_out, end_dim=-2), batch["unit"]["label"].view(-1))
            total_loss = (self.ALPHA * link_loss) + (self.BETA * label_loss) + ( (1 - self.ALPHA- self.BETA) * link_label_loss) 

            output.add_loss(task="total",       data=total_loss)
            output.add_loss(task="link",        data=link_loss)
            output.add_loss(task="link_label",  data=link_label_loss)
            output.add_loss(task="label",       data=label_loss)


        output.add_preds(task="label",          level="unit", data=label_preds)
        output.add_preds(task="link",           level="unit", data=link_preds)
        output.add_preds(task="link_label",     level="unit", data=link_label_preds)

        # output.add_probs(task="relation", level="unit", data=relation_probs)
        # output.add_probs(task="ac",       level="unit", data=ac_probs)
        # output.add_probs(task="s",       level="unit", data=stance_probs)
        return output

       

    

        # return {    
        #             "loss": {   
        #                         "total": total_loss,
        #                         }, 
        #             "preds": {
        #                         "ac": ac_preds, 
        #                         "relation": stance_preds,
        #                         "stance": stance_preds
        #                     },
        #             "probs": {
        #                         "ac": ac_probs, 
        #                         "relation": relation_probs,
        #                         "stance": stance_probs
        #                     },
        #         }