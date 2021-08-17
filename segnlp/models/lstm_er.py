      
      
      
#pytroch
from segnlp.models.base import PTLBase
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from .base import PTLBase
from segnlp.layer_wrappers import Encoder
from segnlp.layer_wrappers import LinkLabeler
from segnlp.layer_wrappers import Segmenter
from segnlp.layer_wrappers import Reducer
from segnlp import utils


class LSTM_ER(PTLBase):

    #https://www.aclweb.org/anthology/P16-1105.pdf

    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)


    def setup_token_rep(self):
        self.word_lstm = Encoder(    
                        layer = "LSTM", 
                        hyperparams = self.hps.get("LSTM", {}),
                        input_size = self.feature_dims["word_embs"] + self.feature_dims["pos_embs"]
                        )


    def setup_token_clf(self):
        self.segmenter = Segmenter(
                                layer = "BigramSeg",
                                hyperparams = self.hps.get("BigramSeg", {}),
                                input_size = self.word_lstm.output_size,
                                output_size = self.task_dims["seg+label"],
                                )


    def setup_seg_rep(self):
        self.agg = Reducer(
                                layer = "Agg",
                                hyperparams = self.hps.get("Agg", {}),
                                input_size = self.word_lstm.output_size,
                            )

        self.deptreelstm = Reducer(
                                    layer = "DepTreeLSTM",
                                    hyperparams = self.hps.get("DepTreeLSTM", {}),
                                    input_size =    self.agg.output_size 
                                                    + self.feature_dims["deprel_embs"]
                                                    + self.task_dims["seg+label"],
                                )
    

    def setup_link_label_clf(self):
        self.link_labeler = LinkLabeler(
                                    layer = "DirLinkLabeler",
                                    hyperparams = self.hps.get("DirLinkLabeler", {}),
                                    input_size = (self.word_lstm.output_size * 2) + self.deptreelstm.output_size,
                                    output_size = self.task_dims["link_label"],
                                    )
                                    

    def token_rep(self, batch:utils.Input, output:utils.Output):
        # lstm_out = (batch_size, max_nr_tokens, lstm_hidden)
        lstm_out, _ = self.word_lstm(
                                input = [
                                            batch["token"]["word_embs"],
                                            batch["token"]["pos_embs"],
                                            ],
                                lengths = batch["token"]["lengths"]
                                )

        return {
                "lstm_out": lstm_out
                }


    def token_clf(self, batch:utils.Input, output:utils.Output):
        logits, preds = self.segmenter(
                                        input = output.stuff["lstm_out"],
                                        )
        return logits, preds


    def seg_rep(self, batch:utils.Input, output:utils.Output):

        # get the average embedding for each segments 
        seg_embs = self.agg(
                            input = batch["token"]["word_embs"], 
                            lengths = output.get_seg_data()["lengths"], 
                            span_idxs = output.get_seg_data()["span_idxs"],
                            )

        pair_data = output.get_pair_data()


        # We create Non-Directional Pair Embeddings using DepTreeLSTM
        # if we follow the example above we get the embedings for all combination of A,B,C. E.g. embeddings for 
        # (A,A), (A,B), (A,C), (B,B), (B,C)
        tree_pair_embs = self.deptreelstm(    
                                            input = (
                                                        output.stuff["lstm_out"],
                                                        batch["token"]["deprel_embs"],
                                                        output.get_preds("seg+label", one_hot = True),
                                                        ),
                                            roots = batch["token"]["root_idxs"],
                                            deplinks = batch["token"]["dephead"],
                                            token_mask = batch["token"]["mask"],
                                            starts = pair_data["nodir"]["p1_end"],
                                            ends = pair_data["nodir"]["p2_end"], #the last token indexes in each seg
                                            lengths = pair_data["nodir"]["lengths"]
                                            )

        # We then add the non directional pair embeddings to the directional pair representations
        # creating dp´ = [dp; s1,s2] (see paper, page 1109)
        seg_embs_flat = seg_embs[batch["seg"]["mask"].type(torch.bool)]

        p1_embs = seg_embs_flat[pair_data["bidir"]["p1"]]
        p2_embs = seg_embs_flat[pair_data["bidir"]["p2"]]
        tree_pair_embs_bidir = tree_pair_embs[pair_data["bidir"]["id"]]

        pair_embs = torch.cat((p1_embs, p2_embs, tree_pair_embs_bidir), dim=-1)

        return {
                "pair_embs":pair_embs
                }


    def link_label_clf(self, batch:utils.Input, output:utils.Output):

        pair_data = output.get_pair_data()

        # We predict link labels for both directions. Get the dominant pair dir
        # plus roots' probabilities
        logits, preds  = self.link_labeler(
                                            input = output.stuff["pair_embs"],
                                            pair_ids = pair_data["bidir"]["id"],
                                            pair_directions = pair_data["bidir"]["direction"],
                                            pair_sample_ids =  pair_data["bidir"]["sample_id"],
                                            pair_p1 = pair_data["bidir"]["p1"],
                                            pair_p2 =  pair_data["bidir"]["p2"],
                                            )
        
        print(lol)
        return logits, preds 


    def loss(self, batch, output):

        seg_label_loss = self.segmenter.loss(
                                            targets =  batch["token"]["seg+label"],
                                            logits = output.logits["seg+label"],
                                            )

        link_label_loss = self.link_labeler.loss(
                                                targets = batch["token"]["link_label"],
                                                logits = output.logits["link_label"],
                                                #token_preds = output.logits["seg+label"],
                                                #token_targets = batch.logits["seg+label"],
                                                #pair_data = output["pair_data"]
                                                )

        total_loss = seg_label_loss + link_label_loss

        return total_loss





        # if not self.inference:
        #     seg_label_probs = seg_label_output["probs"]
        #     link_label_logits = link_label_outputs["link_label_logits"]

        #     label_loss = self.loss(
        #         torch.log(seg_label_probs).view(-1, self.num_seg_labels),
        #         batch["token"]["seg+label"].view(-1))

        #     link_label_loss = self.loss(
        #         torch.log_softmax(link_label_logits, dim=-1),
        #         link_label_outputs["link_label_target"])

        #     total_loss = label_loss + link_label_loss

        #     output.add_loss(task="total", data=total_loss)
        #     output.add_loss(task="link_label", data=link_label_loss)
        #     output.add_loss(task="seg+label", data=label_loss)

        # output.add_preds(task="seg+label",
        #                  level="token",
        #                  data=seg_label_output["preds"])
        # output.add_preds(task="link",
        #                  level="token",
        #                  data=link_label_outputs["link_preds"])
        # output.add_preds(task="link_label",
        #                  level="token",
        #                  data=link_label_outputs["link_label_preds"])

        # # end_time = time.time()
        # # total_time = end_time - start_time
        # # ll_time = link_label_outputs["times"][0]
        # # pair_total = link_label_outputs["times"][1]
        # # graph_const = link_label_outputs["times"][2]
        # # tree_lstm = link_label_outputs["times"][3]

        # # self.schedule.calc_time(
        # #     [total_time, ll_time, pair_total, graph_const, tree_lstm])

        # return output