      
#pytroch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from segnlp.seg_model import SegModel
from segnlp.utils import Batch
from segnlp import utils


class LSTM_ER(SegModel):

    """
    Paper:
    https://aclanthology.org/P17-1002.pdf

    Model Paper:
    https://www.aclweb.org/anthology/P16-1105.pdf

    original code (links on bottom page):
    https://github.com/UKPLab/acl2017-neural_end2end_AM

    """

    def __init__(self,  *args, **kwargs) -> None:   
        super().__init__(*args, **kwargs)


        self.word_emb = self.add_token_embedder(
                                layer = "PretrainedEmbs",
                                hyperparamaters = self.hps.get("word_embs", {})
        )

        self.pos_embs = self.add_token_embedder(
                                    layer  = "Embs",
                                    hyperparamaters = self.hps.get("pos_embs", {}),
        )
    
        self.dep_embs = self.add_token_embedder(
                                    layer  = "Embs",
                                    hyperparamaters = self.hps.get("dep_embs", {}),
        )

        self.dropout  = self.add_token_dropout(
                                                layer = nn.Dropout,
                                                hyperparamaters = self.hps.get("dropout", {}),
                                                )

        self.word_lstm = self.add_token_encoder(    
                                layer = "LSTM", 
                                hyperparamaters = self.hps.get("LSTM", {}),
                                input_size = self.word_emb.output_size  
                                             + self.pos_embs.output_size,
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
                                    input_size =  self.agg.output_size
                                                + self.dep_embs.output_size
                                                + self.task_dims["seg+label"],
                                )

        self.linear_pair_enc = self.add_seg_encoder(
                                layer = "Linear",
                                hyperparamaters = self.hps.get("LinearPairEnc", {}),
                                input_size = (self.word_lstm.output_size * 2) + self.deptreelstm.output_size,
        )

        self.output_dropout  = self.add_seg_dropout(
                                                layer = "Dropout",
                                                hyperparamaters = self.hps.get("output_dropout", {}),
                                                )

        self.link_labeler = self.add_link_labeler(
                                    layer = "DirLinkLabeler",
                                    hyperparamaters = self.hps.get("DirLinkLabeler", {}),
                                    input_size = self.linear_pair_enc.output_size,
                                    output_size = self.task_dims["link_label"],
                                    )


    def token_rep(self, batch: Batch) -> dict:

        # embedd words
        word_embs = self.word_emb(
                            input  = batch.get("token", "str"),
                            lengths = batch.get("token", "lengths"),
                            device = batch.device
        )

        # create pos onehots
        pos_embs = self.pos_embs(
                        input = batch.get("token", "pos"),
                        lengths = batch.get("token", "lengths"),
                        device = batch.device
        )

        #concatenate embeddings
        cat_embs = torch.cat([word_embs, pos_embs], dim = -1)

        # dropout
        cat_embs = self.dropout(cat_embs)

        # pass to lstm
        lstm_out, _ = self.word_lstm(
                                input = cat_embs,
                                lengths = batch.get("token", "lengths")
                                )


        #create dependecy relation onehots
        dep_embs = self.dep_embs(
                        input = batch.get("token", "deprel"),
                        lengths = batch.get("token", "lengths"),
                        device = batch.device
        )
        #apply dropout
        dep_embs = self.dropout(dep_embs)


        return {
                "lstm_out": lstm_out,
                "dep_embs": dep_embs
                }


    def token_clf(self, batch: Batch, rep_out: dict) -> dict:
        logits, preds  = self.segmenter(
                                input = rep_out["lstm_out"],
                                device = batch.device
                                )
        
        batch.add("token", "seg+label", preds)

        return {
                "seg+label": logits,
                }


    def token_loss(self, batch: Batch, clf_out: dict) -> Tensor:
        return self.segmenter.loss(
                                targets =  batch.get("token", "seg+label"),
                                logits = clf_out["seg+label"],
                                )


    def seg_rep(self, batch: Batch, token_rep_out: dict) -> dict:
        
        # if we do not have any candidate pairs, we skip the whole seg_module
        if not len(batch.get("pair", "id")):
            return None
            

        # create label embeddings from predictions of segmentation layer
        token_label_one_hots = utils.one_hot(
                                            batch.get("token", "seg+label", pred = True), 
                                            batch.get("token", "mask"), 
                                            num_classes=self.task_dims["seg+label"]
                                            )

        #apply dropout
        token_label_one_hots = self.dropout(token_label_one_hots.type(torch.float))


        # create new token embeddings
        token_embs = torch.cat([
                                token_label_one_hots,
                                token_rep_out["lstm_out"],
                                token_rep_out["dep_embs"],
                                ],
                                dim = -1
                                )


        # We create Non-Directional Pair Embeddings using DepTreeLSTM
        # If we have the segments A,B,C. E.g. embeddings for the pairs (A,A), (A,B), (A,C), (B,B), (B,C)
        tree_pair_embs = self.deptreelstm(    
                                            input = token_embs,
                                            #roots = batch.get("token", "root_idx"),
                                            deplinks = batch.get("token", "dephead"),
                                            token_mask = batch.get("token", "mask"),
                                            starts = batch.get("pair", "p1_end", bidir = True),
                                            ends = batch.get("pair", "p2_end", bidir = True), #the last token indexes in each seg
                                            lengths = batch.get("pair", "lengths", bidir = True),
                                            device = batch.device
                                            )


        # get the average embedding for each segments 
        seg_embs = self.agg(
                            input = token_rep_out["lstm_out"],
                            lengths = batch.get("seg", "lengths", pred = True), 
                            span_idxs = batch.get("seg", "span_idxs", pred = True),
                            device = batch.device,
                            )

        # We then add the non directional pair embeddings to the directional pair representations
        # creating dp´ = [dp; s1, s2] (see paper above, page 1109)
        seg_embs_flat = seg_embs[batch.get("seg", "mask", pred = True).type(torch.bool)]

        #create bidirectional pair embeddings 
        pair_embs = torch.cat((
                                seg_embs_flat[batch.get("pair", "p1", bidir = True)],
                                seg_embs_flat[batch.get("pair", "p2", bidir = True)],
                                tree_pair_embs
                                ),
                                dim=-1
                                )
    
        # pass embeddings to a final linear layerr
        pair_embs = self.linear_pair_enc(pair_embs)

        # lastly, some tasty dropout
        pair_embs = self.output_dropout(pair_embs)

        return {
                "pair_embs":pair_embs
                }


    def seg_clf(self, batch: Batch, rep_out: dict) -> dict:
    
         # if we do not have any candidate pairs, we skip the whole seg_module
        if rep_out is None:
            return None

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
        
         # if we do not have any candidate pairs, we skip the whole seg_module
        if clf_out is None:
            return None

        return self.link_labeler.loss(
                                                targets = batch.get("pair", "link_label", bidir = True),
                                                logits = clf_out["link_label"],
                                                directions = batch.get("pair", "direction", bidir = True),
                                                true_link = batch.get("pair", "true_link", bidir = True), 
                                                p1_match_ratio = batch.get("pair", "p1-ratio", bidir = True), #pair_data["bidir"]["p1-ratio"],
                                                p2_match_ratio = batch.get("pair", "p2-ratio", bidir = True), #pair_data["bidir"]["p2-ratio"]
                                                )
