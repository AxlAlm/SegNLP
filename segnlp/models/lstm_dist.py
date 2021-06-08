





#basics
from segnlp.layer_wrappers.layer_wrappers import Embedder, Reducer
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#segnlp
from segnlp.layer_wrappers import Encoder
from segnlp.layer_wrappers import Linker
from segnlp.layer_wrappers import Reducer
from segnlp.layer_wrappers import Embedder
from segnlp.layer_wrappers import Labeler
from segnlp.layer_wrappers import LinkLabeler


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

    """

    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.bow = Embedder(
                                layer = "BOW",
                                hyperparams = self.hps.get("BOW", {}),
                            )


        self.word_lstm = Encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = self.feature_dims["word_embs"],
                                    )

        self.minus_span = Reducer(
                                layer = "MinusSpan",
                                hyperparams = self.hps.get("MinusSPan", {}),
                                input_size = self.word_lstm.output_size,
                            )


        self.am_lstm = Encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = self.agg.output_size,
                                    )


        self.ac_lstm = Encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = self.agg.output_size,
                                    )


        input_size = (self.am_lstm.output_size * 2) + self.bow.output_size + self.feature_dims["doc_embs"]
        self.am_ac_lstm =  Encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = input_size
                                    )

        self.last_lstm =  Encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("LSTM", {}),
                                    input_size = self.am_ac_lstm.output_size,
                                    )

        self.linker = Linker(
                                layer = "PairingLayer",
                                hyperparams = self.hps.get("PairingLayer", {}),
                                input_size = self.am_ac_lstm.output_size,
                                )

        self.labeler = LinkLabeler(
                                        layer = "LinearCLF",
                                        hyperparams = self.hps.get("LinearCLF", {}),
                                        input_size = self.am_ac_lstm.output_size,
                                        )

        self.link_labeler = LinkLabeler(
                                        layer = "LinearCLF",
                                        hyperparams = self.hps.get("LinearCLF", {}),
                                        input_size = self.am_ac_lstm.output_size,
                                        )


    @classmethod
    def name(self):
        return "LSTM_DIST"


    def forward(self, batch, output):
        
        # batch is sorted by length of prediction level which is seg
        # so we need to sort the word embeddings for the sample, pass to lstm then return to 
        # original order
        sorted_lengths_tok, sorted_indices = torch.sort(batch["token"]["lengths"], descending=True)
        _ , original_indices = torch.sort(sorted_indices, descending=False)

        # input (Batch_dize, nr_tokens, word_emb_dim)
        # output (Batch_dize, nr_tokens, word_emb_dim)
        lstm_out, _ = self.word_lstm(
                                        input = batch["token"]["word_embs"][sorted_indices], 
                                        lengths = sorted_lengths_tok
                                    )
        lstm_out = lstm_out[original_indices]

        # create span representation for Argument Components and Argumentative Markers
        am_minus_embs = self.minus_span(
                                        input = lstm_out, 
                                        span_idxs = batch["am"]["span_idxs"]
                                        )
        ac_minus_embs = self.minus_span(lstm_out, batch["seg"]["span_idxs"])

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


        # create BOW features
        bow = self.bow(
                        word_encs = batch["token"]["words"], 
                        span_idxs = batch["seg"]["span_idxs"]
                        )

        # concatenate the output from Argument Component BiLSTM and Argument Marker BiLSTM with BOW and with structural features stored in "doc_embs"
        cat_emb = torch.cat((am_lstm_out, ac_lstm_out, bow, batch["seg"]["doc_embs"]), dim=-1)
        adu_emb, _= self.am_ac_lstm(cat_emb, batch["unit"]["lengths"])

  
        # Classification of label and link labels is pretty straight forward
        link_label_logits, link_label_preds = self.link_labeler(adu_emb)
        label_logits, label_preds = self.labeler(adu_emb)
        
        adu_emb, _ = self.last_lstm(adu_emb, batch["unit"]["lengths"])
        link_logits, link_preds = self.linker(adu_emb, unit_mask=batch["unit"]["mask"])


        return {
                "logits": {
                            "link": link_logits,
                            "label": label_logits,
                            "link_label": link_label_logits,
                            },
                "preds":{
                            "link":link_preds,
                            "label": label_preds,
                            "link_label": link_label_preds,
                            },

                }


    def loss(self, batch, forward_output:dict):


        link_loss = self.linker.loss(
                                torch.flatten(forward_output["logits"]["link"], end_dim=-2), 
                                batch["unit"]["link"].view(-1)
                                )

        link_label_loss = self.link_labeler.loss(
                                    torch.flatten(forward_output["logits"]["link_label"], end_dim=-2),
                                     batch["unit"]["link_label"].view(-1)
                                     )

        label_loss = self.labeler.loss(
                                torch.flatten(forward_output["logits"]["label"], end_dim=-2), 
                                batch["unit"]["label"].view(-1)
                                )

        ## this is the reported loss aggregation in the paper, but...
        #total_loss = -((self.loss_weight * link_loss) + (self.loss_weight * label_loss) + ( (1 - self.loss_weight- self.loss_weight) * link_label_loss))
        ## in the code the loss is different
        ## https://github.com/kuribayashi4/span_based_argumentation_parser/blob/614343b18e7d98293a2b020f9ab05b86355e18df/src/classifier/parsing_loss.py#L88-L91
        tw = self.hps["general"]["task_weight"]
        total_loss = ((1 - tw - tw) * link_loss) - (tw * label_loss) + (tw * link_label_loss)


