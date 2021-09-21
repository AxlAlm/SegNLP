





#basics
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#segnlp
from .base import BaseModel
from segnlp import utils

class LSTM_DIST(BaseModel):

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


        self.word_lstm = self.add_token_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("Word_LSTM", {}),
                                    input_size = self.feature_dims["word_embs"],
                                    )

        self.minus_span = self.add_seg_rep(
                                        layer = "MinusSpan",
                                        hyperparams = self.hps.get("MinusSPan", {}),
                                        input_size = self.word_lstm.output_size,
                                    )

        self.am_lstm = self.add_seg_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("AM_LSTM", {}),
                                    input_size = self.minus_span.output_size,
                                    )

        self.ac_lstm = self.add_seg_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("AC_LSTM", {}),
                                    input_size = self.minus_span.output_size,
                                    )
                                
        self.seg_bow = self.add_seg_embedder(
                                        layer = "SegBOW",
                                        hyperparams = self.hps.get("SegBOW", {}),
                                        )

        self.seg_pos = self.add_seg_embedder(
                                        layer = "SegPos",
                                        hyperparams = {},
                                        )

        self.bow_dim_redu  = self.add_encoder(
                                                layer = "Linear",
                                                hyperparams = self.hps.get("BOW_dim_redu", {}),
                                                input_size = self.seg_bow.output_size,
                                                module = "segment_module"
        )


        input_size = (self.am_lstm.output_size * 2) + self.seg_bow.output_size + self.seg_pos.output_size
        self.adu_lstm =  self.add_seg_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("ADU_LSTM", {}),
                                    input_size = input_size,
                                    )

        self.link_lstm =  self.add_seg_encoder(
                                    layer = "LSTM",
                                    hyperparams = self.hps.get("Link_LSTM", {}),
                                    input_size = self.adu_lstm.output_size,
                                    )


        self.pairer = self.add_pair_rep(
                                layer = "Pairer",
                                hyperparams = self.hps.get("Pairer", {}),
                                input_size = self.link_lstm.output_size,
                                
                                )

        self.linker = self.add_linker(
                                layer = "PairCLF",
                                hyperparams = self.hps.get("PairCLF", {}),
                                input_size = self.pairer.output_size,
                                )

        self.labeler = self.add_labeler(
                                        layer = "LinearCLF",
                                        hyperparams = self.hps.get("LinearCLF", {}),
                                        input_size = self.adu_lstm.output_size,
                                        output_size = self.task_dims["label"],

                                        )

        self.link_labeler = self.add_link_labeler(
                                            layer = "LinearCLF",
                                            hyperparams = self.hps.get("LinearCLF", {}),
                                            input_size = self.adu_lstm.output_size,
                                            output_size = self.task_dims["link_label"],
                                            )


    @classmethod
    def name(self):
        return "LSTM_DIST"


    def seg_rep(self, batch: utils.BatchInput, output: utils.BatchOutput):


        lstm_out, _ = self.word_lstm(
                                        input = batch.get("token", "word_embs"), 
                                        lengths =  batch.get("token", "lengths"),
                                    )


        # create span representation for Argument Components and Argumentative Markers
        am_minus_embs = self.minus_span(
                                        input = lstm_out, 
                                        span_idxs = batch.get("am", "span_idxs"),
                                        device = batch.device
                                        )
        ac_minus_embs = self.minus_span(
                                        input = lstm_out, 
                                        span_idxs = batch.get("seg", "span_idxs"),
                                        device = batch.device
                                        )
        
        # pass the minus representation for each type of segment to seperate LSTMs
        am_lstm_out, _ = self.am_lstm(
                                    am_minus_embs, 
                                    batch.get("am", "lengths"),
                                    )
        ac_lstm_out, _ = self.ac_lstm(
                                    ac_minus_embs,
                                    batch.get("seg", "lengths"),
                                    )

        # The BOW features, φ(wi:j), are create from one hot encodings + positional features (see below). 
        # As we are replicating the ELMO version it doesnt include the aggregated word embeddings.
        # NOTE! It seems like the author are concatenating one-hot encodings of words to a large vector for each
        # span; (number_spans, max_tokens, vocab_size) -> (number_spans, max_tokens * vocab_size)
        # as this transformation will create huge parse representation for each segment we create one-hot encodings
        # for each segment by adding them instead of concatenating (see segnlp.layers.embedders.seg_bow.SegBOW)
        # i.e. a vector of word counts
        seg_bow = self.seg_bow(
                        input = batch.get("token", "str"), 
                        lengths = batch.get("token", "lengths"),
                        span_idxs = batch.get("adu", "span_idxs"),
                        device = batch.device
                        )
        
        #we reduce the dim
        seg_bow = self.bow_dim_redu(seg_bow)

        # positional features for segments
        segpos = self.seg_pos(
                            document_paragraph_id = batch.get("seg", "document_paragraph_id"), 
                            nr_paragraphs_doc = batch.get("seg", "nr_paragraphs_doc"),
                            lengths = batch.get("seg", "lengths"),
                            device = batch.device

                            )
    

        # concatenate the output from Argument Component BiLSTM and Argument Marker BiLSTM with BOW and with structural features stored in "doc_embs"
        cat_emb = torch.cat((
                                am_lstm_out, 
                                ac_lstm_out, 
                                seg_bow, 
                                segpos,
                                ), dim=-1)
            
    
        # run concatenated features through and LSTM, output will be used to predict label and link_label
        adu_emb, _ = self.adu_lstm(cat_emb, batch.get("seg", "lengths"))

        #then run the adu_embs through a final lstm to create features for linking.
        adu_emb_link, _ = self.link_lstm(adu_emb, batch.get("seg", "lengths"))

        # create embeddings for all pairs
        pair_embs = self.pairer(
                                adu_emb_link,
                                device = batch.device
                                )

        return {
                "adu_emb": adu_emb,
                "pair_embs": pair_embs
                }


    def seg_clf(self, batch: utils.BatchInput, output: utils.BatchOutput):

        # classify the label of the segments
        label_logits, label_preds = self.labeler(output.stuff["adu_emb"])

        # classify the label of the links
        link_label_logits, link_label_preds = self.link_labeler(output.stuff["adu_emb"])

        # classify links
        link_logits, link_preds = self.linker(
                                input = output.stuff["pair_embs"], 
                                segment_mask = batch.get("seg", "mask")
                                )
            
        return [
                {
                    "task": "label",
                    "logits": label_logits,
                    "preds": label_preds,
                  },
                {
                    "task": "link_label",
                    "logits": link_label_logits,
                    "preds": link_label_preds,
                  },
                {
                    "task": "link",
                    "logits": link_logits,
                    "preds": link_preds,
                  }
                ]
                    






    def loss(self, batch: utils.BatchInput, output: utils.BatchOutput):

        link_loss = self.linker.loss(
                                torch.flatten(output.logits["link"], end_dim=-2), 
                                batch.get("seg", "link").view(-1)
                                )

        link_label_loss = self.link_labeler.loss(
                                    torch.flatten(output.logits["link_label"], end_dim=-2),
                                     batch.get("seg", "link_label").view(-1)
                                     )

        label_loss = self.labeler.loss(
                                torch.flatten(output.logits["label"], end_dim=-2), 
                                batch.get("seg", "label").view(-1)
                                )

        ## this is the reported loss aggregation in the paper, but...
        #total_loss = -((self.loss_weight * link_loss) + (self.loss_weight * label_loss) + ( (1 - self.loss_weight- self.loss_weight) * link_label_loss))
        ## in the code the loss is different
        ## https://github.com/kuribayashi4/span_based_argumentation_parser/blob/614343b18e7d98293a2b020f9ab05b86355e18df/src/classifier/parsing_loss.py#L88-L91
        tw = self.hps["general"]["task_weight"]
        total_loss = ((1 - tw - tw) * link_loss) + (tw * label_loss) + (tw * link_label_loss)

        return total_loss
