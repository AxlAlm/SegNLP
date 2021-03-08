

#basics
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#hotam
from hotam.nn.layers.lstm import LSTM_LAYER
import hotam.utils as u


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

            Link/relation: 

    """

    def __init__(self,  hyperparamaters:dict, task_dims:dict, feature_dims:dict):
        super().__init__()

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

        self.word_lstm = LSTM_LAYER(  
                                input_size = self.WORD_FEATURE_DIM,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                )


        self.am_lstm = LSTM_LAYER(  
                                input_size = self.HIDDEN_DIM*3,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                )

        self.ac_lstm = LSTM_LAYER(  
                                input_size = self.HIDDEN_DIM*3,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                )

        # input to adu_lstm is the am + ac + doc features 
        adu_shape = ((self.HIDDEN_DIM*(2 if self.BI_DIR else 1)) * 2 ) + self.DOC_FEATURE_DIM

        self.adu_lstm = LSTM_LAYER(  
                                    input_size = adu_shape,
                                    hidden_size=self.HIDDEN_DIM,
                                    num_layers= self.NUM_LAYERS,
                                    bidirectional=self.BI_DIR,
                                    )
        
        #self.max_rel = len(task2labels["relation"])
        #self.relation_layer = nn.Linear(self.HIDDEN_DIM*(2 if self.BI_DIR else 1), MAX_NR_AC)


        self.type_rep_size*3 + self.relative_position_info_size
        self.relation_clf = nn.Linear(self.HIDDEN_DIM*(2 if self.BI_DIR else 1), 1)



        self.stance_clf = nn.Linear(self.HIDDEN_DIM*(2 if self.BI_DIR else 1), task_dims["stance"])
        self.ac_clf = nn.Linear(self.HIDDEN_DIM*(2 if self.BI_DIR else 1), task_dims["ac"])
    
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

        minus_reps = torch.zeros((batch_size, nr_seq, feature_dim))
        
        for idx in range(batch_size):
            for sidx,(i,j) in enumerate(spans[idx]):


                if i==0 and j == 0:
                    continue

                if i-1 == -1:
                    f_pre = torch.zeros(hidden_dim)
                else:
                    f_pre = forward[idx][i-1]


                if j+1 > backward.shape[1]:
                    b_post = torch.zeros(hidden_dim)
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


    def forward(self, Input, Output):

        word_embs = batch["word_embs"] 
        lengths_tok  = batch["lengths_tok"]
        lengths_seq = batch["lengths_seq"]
        ac_mask = batch["ac_mask"]
        batch_size = ac_mask.shape[0]
        max_nr_acs = ac_mask.shape[1]

        # Wi:j 
        W = batch["doc_embs"]

        # batch is sorted by length of prediction level which is Argument Components
        # so we need to sort the word embeddings for the sample, pass to lstm then return to 
        # original order
        sorted_lengths_tok, sorted_indices = torch.sort(lengths_tok, descending=True)
        _ , original_indices = torch.sort(sorted_indices, descending=False)

        # 1
        # input (Batch_dize, nr_tokens, word_emb_dim)
        # output (Batch_dize, nr_tokens, word_emb_dim)
        lstm_out, _ = self.word_lstm(word_embs[sorted_indices], sorted_lengths_tok)
        lstm_out = lstm_out[original_indices]

        # 2
        # create span representation for Argument Components and Argumentative Markers
        am_minus_embs = self.__minus_span(lstm_out, batch["am_spans"])
        ac_minus_embs = self.__minus_span(lstm_out, batch["ac_spans"])

        # 3
        # pass each of the spans to a seperate BiLSTM. 
        am_lstm_out, _ = self.am_lstm(am_minus_embs, lengths_seq)
        ac_lstm_out, _ = self.ac_lstm(ac_minus_embs, lengths_seq)

        # 4
        # concatenate the output from Argument Component BiLSTM and Argument Marker BiLSTM with BOW embeddigns W
        contex_emb = torch.cat((am_lstm_out, ac_lstm_out, W), dim=-1)

        # 5
        #final_out, _ = self.adu_lstm(contex_emb, lengths_seq)

        # 6
        # Classification of AC and Stance is pretty straight forward
        stance_out = self.stance_clf(contex_emb)
        ac_out = self.ac_clf(contex_emb)



        # input = (batch_size, max_units, max_units, contex_emb.shape[-1]*3 + max_units)
        #
        # linkCLF(input)
        # output = (batch_size, max_units, max_units)
        # max_spanning_tree(output)



 
        #
        # 1) create this matrix = Hj ; Hi ; Hj * Hi ; pos_encs
        # (batch_size, max_units, emb_size)
        #
        # 2) pass to softmax ->
        # 

        # Classification of Links between ACs is done by concatenate a one-hot vector representing positions
        # with the span representations from final_out.
        # 
        # maximum spanning tree algorithm
        # relation_out_new = torch.zeros((*ac_mask.shape, ac_mask.shape[-1]))
        # relation_probs = torch.zeros((*ac_mask.shape, ac_mask.shape[-1]))
        # print("BATCH SIZE", batch_size)
        # for i in range(batch_size):
        #     # print(i)
        #     sample_mask = ac_mask[i]
        #     sample_m_mask = sample_mask.repeat(max_nr_acs,1)
        #     # print(ac_mask[i])
        #     # print(sample_m_mask)
        #     # print(relation_out[i,:, :6].shape, ac_mask.shape)
        #     out = relation_out[i,:, :max_nr_acs] * sample_m_mask
            
        #     out_copy = out.detach().clone()
        #     sample_m_mask = sample_m_mask.type(torch.bool)
        #     out_copy[~sample_m_mask] = float('-inf')
        #     # print("INF OUT", out_copy)
        #     out_softmax = F.softmax(out_copy, dim=-1)

        #     # print("SOFTMAX", out_softmax)
        #     # print("OUT", out)
        #     relation_probs[i] = out_softmax
        #     relation_out_new[i] = out
        #relation_out = relation_out_new


        stance_probs = F.softmax(stance_out, dim=-1)
        ac_probs = F.softmax(ac_out, dim=-1)

        relations_preds = torch.argmax(relation_out, dim=-1)
        stance_preds = torch.argmax(stance_out, dim=-1)
        ac_preds = torch.argmax(ac_out, dim=-1)


        # we want to ignore -1  in the loss function so we set pad_values to -1, default is 0
        batch.change_pad_value(-1)
        
        relation_loss = self.loss(torch.flatten(relation_out, end_dim=-2), batch["relation"].view(-1))
        stance_loss = self.loss(torch.flatten(stance_out, end_dim=-2), batch["stance"].view(-1))
        ac_loss = self.loss(torch.flatten(ac_out, end_dim=-2), batch["ac"].view(-1))
        total_loss = (self.ALPHA * relation_loss) + (self.BETA * stance_loss) + ( (1 - (self.ALPHA-self.BETA)) * ac_loss) 


        output.add_loss(task="total",    data=total_loss)
        output.add_loss(task="relation", data=relation_loss)
        output.add_loss(task="ac",       data=ac_loss)
        output.add_loss(task="stance",   data=stance_loss)

        output.add_preds(task="relation", level="unit", data=relations_preds)
        output.add_preds(task="ac",       level="unit", data=ac_preds)
        output.add_preds(task="stance",   level="unit", data=stance_preds)

        output.add_probs(task="relation", level="unit", data=relation_probs)
        output.add_probs(task="ac",       level="unit", data=ac_probs)



        prediction_levels = ["token", "unit", "span"]

        output.add_prebs(task="link_label",   level="unit", data=link_label_pred)
        output.add_prebs(task="link",  level="unit", data=link_pred)
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