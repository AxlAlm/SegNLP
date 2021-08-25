





#basics
from torch.nn.modules import module
from segnlp.layer_wrappers.layer_wrappers import Embedder, Reducer
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#segnlp
from .base import PTLBase
from segnlp import utils

class LSTM_DIST(PTLBase):

    """

    Implementation of an LSTM-Minus-based span representation network
    for Argument Structre Parsing / Argument Mining

    paper:
    https://www.aclweb.org/anthology/P19-1464/

    Original code:
    https://github.com/kuribayashi4/span_based_argumentation_parser/tree/614343b18e7d98293a2b020f9ab05b86355e18df


    More on LSTM-Minus
    https://www.aclweb.org/anthology/P16-1218/

    """

    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.bow = self.add_embedder(
                                layer = "BOW",
                                hyperparams = self.hps.get("BOW", {}),
                                module = "segment_module"
                            )


        self.word_lstm = self.add_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = self.feature_dims["word_embs"],
                                    module = "segment_module"
                                    )

        self.minus_span = self.add_reducer(
                                layer = "MinusSpan",
                                hyperparams = self.hps.get("MinusSPan", {}),
                                input_size = self.word_lstm.output_size,
                                module = "segment_module"
                            )

        self.am_lstm = self.add_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = self.minus_span.output_size,
                                    module = "segment_module"
                                    )


        self.ac_lstm = self.add_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = self.minus_span.output_size,
                                    module = "segment_module"
                                    )


        input_size = (self.am_lstm.output_size * 2) + self.bow.output_size + self.feature_dims["doc_embs"]
        self.am_ac_lstm =  self.add_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = input_size,
                                    module = "segment_module"
                                    )

        self.last_lstm =  self.add_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = self.am_ac_lstm.output_size,
                                    module = "segment_module"
                                    )

        self.linker = self.add_linker(
                                layer = "Pairer",
                                hyperparams = self.hps.get("Pairer", {}),
                                input_size = self.am_ac_lstm.output_size,
                                )

        self.labeler = self.add_labeler(
                                        layer = "LinearCLF",
                                        hyperparams = self.hps.get("LinearCLF", {}),
                                        input_size = self.am_ac_lstm.output_size,
                                        output_size = self.task_dims["label"],

                                        )

        self.link_labeler = self.add_link_labeler(
                                            layer = "LinearCLF",
                                            hyperparams = self.hps.get("LinearCLF", {}),
                                            input_size = self.am_ac_lstm.output_size,
                                            output_size = self.task_dims["label"],
                                            )


    @classmethod
    def name(self):
        return "LSTM_DIST"


    def seg_rep(self, batch:utils.Input, output:utils.Output):

        lstm_out, _ = self.word_lstm(
                                        input = batch["token"]["word_embs"], 
                                        lengths =  batch["token"]["lengths"]
                                    )

        # create span representation for Argument Components and Argumentative Markers
        am_minus_embs = self.minus_span(
                                        input = lstm_out, 
                                        span_idxs = batch["am"]["span_idxs"]
                                        )
        ac_minus_embs = self.minus_span(
                                        input = lstm_out, 
                                        span_idxs = batch["seg"]["span_idxs"]
                                        )
        
        # pass the minus representation for each type of segment to seperate LSTMs
        am_lstm_out, _ = self.am_lstm(am_minus_embs, batch["am"]["lengths"])
        ac_lstm_out, _ = self.ac_lstm(ac_minus_embs, batch["seg"]["lengths"])


        # create BOW features
        bow = self.bow(
                        word_encs = batch["token"]["words"], 
                        span_idxs = batch["seg"]["span_idxs"]
                        )

        # concatenate the output from Argument Component BiLSTM and Argument Marker BiLSTM with BOW and with structural features stored in "doc_embs"
        cat_emb = torch.cat((am_lstm_out, ac_lstm_out, bow, batch["seg"]["doc_embs"]), dim=-1)

        # run concatenated features through and LSTM, output will be used to predict label and link_label
        adu_emb, _ = self.am_ac_lstm(cat_emb, batch["seg"]["lengths"])

        #then run the adu_embs through a final lstm to create features for linking.
        adu_emb_last, _ = self.last_lstm(adu_emb, batch["seg"]["lengths"])


        return {
                "adu_emb": adu_emb,
                "adu_emb_last": adu_emb_last
                }


    def seg_clf(self, batch:utils.Input, output:utils.Output):

        label_outs = self.labeler(output.stuff["adu_emb"])
        link_label_outs = self.link_labeler(output.stuff["adu_emb"])
        link_outs = self.linker(
                                input = output.stuff["adu_emb_last"], 
                                segment_mask = batch["seg"]["mask"]
                                )
            
        return label_outs + link_outs + link_label_outs




    def loss(self,  batch:utils.Input, output:utils.Output):


        link_loss = self.linker.loss(
                                torch.flatten(output.logits["link"], end_dim=-2), 
                                batch["seg"]["link"].view(-1)
                                )

        link_label_loss = self.link_labeler.loss(
                                    torch.flatten(output.logits["link_label"], end_dim=-2),
                                     batch["seg"]["link_label"].view(-1)
                                     )

        label_loss = self.labeler.loss(
                                torch.flatten(output.logits["label"], end_dim=-2), 
                                batch["seg"]["label"].view(-1)
                                )

        ## this is the reported loss aggregation in the paper, but...
        #total_loss = -((self.loss_weight * link_loss) + (self.loss_weight * label_loss) + ( (1 - self.loss_weight- self.loss_weight) * link_label_loss))
        ## in the code the loss is different
        ## https://github.com/kuribayashi4/span_based_argumentation_parser/blob/614343b18e7d98293a2b020f9ab05b86355e18df/src/classifier/parsing_loss.py#L88-L91
        tw = self.hps["general"]["task_weight"]
        total_loss = ((1 - tw - tw) * link_loss) - (tw * label_loss) + (tw * link_label_loss)


