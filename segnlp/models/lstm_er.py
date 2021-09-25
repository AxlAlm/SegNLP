      
      
      
#pytroch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from .base import BaseModel
from segnlp import utils
from segnlp.utils import Batch


class LSTM_ER(BaseModel):

    """
    
    Paper:
    https://www.aclweb.org/anthology/P16-1105.pdf

    original code (links on bottom page):
    https://github.com/UKPLab/acl2017-neural_end2end_AM

    """

    def __init__(self,  *args, **kwargs) -> None:   
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


    def token_rep(self, batch: Batch) -> dict:

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


    def token_clf(self, batch: Batch, rep_out: dict) -> dict:
        logits, preds  = self.segmenter(
                                input = rep_out["lstm_out"],
                                )
        
        batch.add("token", "seg+label", preds)
        return {"seg+label": logits}


    def token_loss(self, batch: Batch, clf_out: dict) -> Tensor:
        return self.segmenter.loss(
                                targets =  batch.get("token", "seg+label"),
                                logits = clf_out["seg+label"],
                                )

    @utils.timer
    def seg_rep(self, batch: Batch, token_rep_out: dict) -> dict:

        # get the average embedding for each segments 
        seg_embs = self.agg(
                            input = batch.get("token", "embs"),
                            lengths = batch.get("seg", "lengths", pred = True), 
                            span_idxs = batch.get("seg", "span_idxs", pred = True),
                            device = batch.device,
                            )

        #create dependecy relation onehots
        dep_one_hots = self.dep_onehot(
                        input = batch.get("token", "deprel"),
                        lengths = batch.get("token", "lengths"),
                        device = batch.device
        )

        token_label_one_hots = utils.one_hot(
                                            batch.get("token", "seg+label", pred = True), 
                                            batch.get("token", "mask"), 
                                            num_classes=self.task_dims["seg+label"]
                                            )

        # We create Non-Directional Pair Embeddings using DepTreeLSTM
        # If we have the segments A,B,C. E.g. embeddings for the pairs (A,A), (A,B), (A,C), (B,B), (B,C)
        tree_pair_embs = self.deptreelstm(    
                                            input = (
                                                        token_rep_out["lstm_out"],
                                                        dep_one_hots,
                                                        token_label_one_hots,
                                                        ),
                                            roots = batch.get("token", "root_idx"),
                                            deplinks = batch.get("token", "dephead"),
                                            token_mask = batch.get("token", "mask"),
                                            starts = batch.get("pair", "p1_end", bidir = False),
                                            ends = batch.get("pair", "p2_end", bidir = False), #the last token indexes in each seg
                                            lengths = batch.get("pair", "lengths", bidir = False),
                                            device = batch.device
                                            )

        # We then add the non directional pair embeddings to the directional pair representations
        # creating dp´ = [dp; s1, s2] (see paper above, page 1109)
        seg_embs_flat = seg_embs[batch.get("seg", "mask", pred = True).type(torch.bool)]

        pair_embs = torch.cat((
                                seg_embs_flat[batch.get("pair", "p1", bidir = True)],
                                seg_embs_flat[batch.get("pair", "p2", bidir = True)],
                                tree_pair_embs[batch.get("pair", "id", bidir = True, pred = True)]
                                ),
                                dim=-1
                                )
    

        pair_embs = self.linear_pair_enc(pair_embs)

        return {
                "pair_embs":pair_embs
                }


    def seg_clf(self, batch: Batch, rep_out: dict) -> dict:

        # We predict link labels for both directions. Get the dominant pair dir
        # plus roots' probabilities
        link_label_logits, link_label_preds, link_preds =  self.link_labeler(
                                            input = rep_out["pair_embs"],
                                            pair_p1 = batch.get("pair", "p1", bidir = True),
                                            pair_p2 =  batch.get("pair", "p2", bidir = True),
                                            sample_id = batch.get("pair", "sample_id", bidir = True)
                                            )

        # add our prediction to the batch
        batch.add("p_seg", "link", link_preds)
        batch.add("p_seg", "link_label", link_label_preds)


        return { "link_label": link_label_logits}
                    

    def seg_loss(self, batch: Batch, clf_out: dict) -> Tensor:
        return self.link_labeler.loss(
                                                targets = batch.get("pair", "link_label", bidir = True),
                                                logits = clf_out["link_label"],
                                                directions = batch.get("pair", "direction", bidir = True),
                                                true_link = batch.get("pair", "true_link", bidir = True), 
                                                p1_match_ratio = batch.get("pair", "p1-ratio", bidir = True), #pair_data["bidir"]["p1-ratio"],
                                                p2_match_ratio = batch.get("pair", "p2-ratio", bidir = True), #pair_data["bidir"]["p2-ratio"]
                                                )
