      
      
      
#pytroch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from .base import BaseModel
from segnlp import utils

class LSTM_ER(BaseModel):

    """
    
    Paper:
    https://www.aclweb.org/anthology/P16-1105.pdf

    original code (links on bottom page):
    https://github.com/UKPLab/acl2017-neural_end2end_AM

    """

    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)


        self.pos_onehot = self.add_token_embedder(
                                    layer  = "PosOneHots",
                                    hyperparamaters = {},
        )

        self.word_lstm = self.add_token_encoder(    
                                layer = "LSTM", 
                                hyperparamaters = self.hps.get("LSTM", {}),
                                input_size = self.feature_dims["word_embs"] + 
                                             self.pos_onehot.output_size,
                                )

        self.dep_onehot = self.add_token_embedder(                                  
                                    layer  = "DepOneHots",
                                    hyperparamaters = {},
        )

        self.segmenter = self.add_segmenter(
                                layer = "BigramSeg",
                                hyperparamaters = self.hps.get("BigramSeg", {}),
                                input_size = self.word_lstm.output_size,
                                output_size = self.task_dims["seg+label"],
                                )

        self.agg = self.add_seg_rep(
                            layer = "Agg",
                            hyperparamaters = self.hps.get("Agg", {}),
                            input_size = self.word_lstm.output_size,
                        )

        self.deptreelstm = self.add_pair_rep(
                                    layer = "DepTreeLSTM",
                                    hyperparamaters = self.hps.get("DepTreeLSTM", {}),
                                    input_size =    self.agg.output_size
                                                    + self.dep_onehot.output_size
                                                    + self.task_dims["seg+label"],
                                )

        self.linear_pair_enc = self.add_seg_encoder(
                                layer = "Linear",
                                hyperparamaters = self.hps.get("LinearPairEnc", {}),
                                input_size = (self.word_lstm.output_size * 2) + self.deptreelstm.output_size,
        )

        self.link_labeler = self.add_link_labeler(
                                    layer = "DirLinkLabeler",
                                    hyperparamaters = self.hps.get("DirLinkLabeler", {}),
                                    input_size = self.linear_pair_enc.output_size,
                                    output_size = self.task_dims["link_label"],
                                    )


    @utils.timer
    def token_rep(self, batch: utils.BatchInput, output: utils.BatchOutput):

        #create pos onehots
        pos_one_hots = self.pos_onehot(
                        input = batch.get("token", "pos"),
                        lengths = batch.get("token", "lengths"),
                        device = batch.device
        )

        # lstm_out = (batch_size, max_nr_tokens, lstm_hidden)
        lstm_out, _ = self.word_lstm(
                                input = [
                                            batch.get("token", "embs"),
                                            pos_one_hots,
                                            ],
                                lengths = batch.get("token", "lengths")
                                )

        return {
                "lstm_out": lstm_out
                }

    @utils.timer
    def token_clf(self, batch: utils.BatchInput, output: utils.BatchOutput):
        logits, preds  = self.segmenter(
                                input = output.stuff["lstm_out"],
                                )
        return [
                {
                "task": self.seg_task,
                "logits": logits,
                "preds": preds,
                }
                ]

    @utils.timer
    def seg_rep(self, batch: utils.BatchInput, output: utils.BatchOutput):


        # get the average embedding for each segments 
        seg_embs = self.agg(
                            input = batch.get("token", "embs"),
                            lengths = output.get_seg_data()["lengths"], 
                            span_idxs = output.get_seg_data()["span_idxs"],
                            device = batch.device
                            )

        pair_data = output.get_pair_data()


        #create dependecy relation onehots
        dep_one_hots = self.dep_onehot(
                        input = batch.get("token", "deprel"),
                        lengths = batch.get("token", "lengths"),
                        device = batch.device
        )

        # We create Non-Directional Pair Embeddings using DepTreeLSTM
        # If we have the segments A,B,C. E.g. embeddings for the pairs (A,A), (A,B), (A,C), (B,B), (B,C)
        tree_pair_embs = self.deptreelstm(    
                                            input = (
                                                        output.stuff["lstm_out"],
                                                        dep_one_hots,
                                                        output.get_preds("seg+label", one_hot = True),
                                                        ),
                                            roots = batch.get("token", "root_idx"),
                                            deplinks = batch.get("token", "dephead"),
                                            token_mask = batch.get("token", "mask"),
                                            starts = pair_data["nodir"]["p1_end"],
                                            ends = pair_data["nodir"]["p2_end"], #the last token indexes in each seg
                                            lengths = pair_data["nodir"]["lengths"],
                                            device = batch.device
                                            )

        # We then add the non directional pair embeddings to the directional pair representations
        # creating dp´ = [dp; s1, s2] (see paper above, page 1109)
        seg_embs_flat = seg_embs[batch.get("seg", "mask").type(torch.bool)]

        pair_embs = torch.cat((
                                seg_embs_flat[pair_data["bidir"]["p1"]], 
                                seg_embs_flat[pair_data["bidir"]["p2"]],
                                tree_pair_embs[pair_data["bidir"]["id"]]
                                ),
                                dim=-1
                                )
    

        pair_embs = self.linear_pair_enc(pair_embs)

        return {
                "pair_embs":pair_embs
                }


    @utils.timer
    def seg_clf(self, batch: utils.BatchInput, output: utils.BatchOutput):

        pair_data = output.get_pair_data()

        # We predict link labels for both directions. Get the dominant pair dir
        # plus roots' probabilities
        link_label_logits, link_label_preds, link_preds =  self.link_labeler(
                                            input = output.stuff["pair_embs"],
                                            pair_p1 = pair_data["bidir"]["p1"],
                                            pair_p2 =  pair_data["bidir"]["p2"],
                                            )

        return [
                {
                    "task": "link_label",
                    "logits": link_label_logits,
                    "preds": link_label_preds,
                    "level": "p_seg",
                  },
                {
                    "task": "link",
                    "preds": link_preds,
                    "level": "p_seg",
                  }
                ]
                    
    
    def token_loss(self, batch: utils.BatchInput, output: utils.BatchOutput):
        return self.segmenter.loss(
                                targets =  batch.get("token", "seg+label"),
                                logits = output.logits["seg+label"],
                                )


    def seg_loss(self, batch: utils.BatchInput, output: utils.BatchOutput):
        pair_data = output.get_pair_data()
        return self.link_labeler.loss(
                                                targets = pair_data["bidir"]["link_label"],
                                                logits = output.logits["link_label"],
                                                directions = pair_data["bidir"]["direction"],
                                                true_link = pair_data["bidir"]["true_link"], 
                                                p1_match_ratio = pair_data["bidir"]["p1-ratio"],
                                                p2_match_ratio = pair_data["bidir"]["p2-ratio"]
                                                )
