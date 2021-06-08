      
      
      
#pytroch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from segnlp.layer_wrappers import Encoder
from segnlp.layer_wrappers import LinkLabeler
from segnlp.layer_wrappers import Segmenter
from segnlp import utils


class LSTM_ER(nn.Module):

    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)


        self.word_lstm = Encoder(    
                                layer = "LSTM", 
                                hyperparams = self.hps.get("LSTM", {}),
                                input_size = self.fc1.output_size
                                )

        self.segmenter = Segmenter(
                                layer = "BigramSeg",
                                hyperparams = self.hps.get("BigramSeg", {}),
                                input_size = self.word_lstm.output_size,
                                output_size = self.task_dims["seg+label"],
                                )


        self.link_labeler = LinkLabeler(
                                    layer = "DepPairingLayer",
                                    hyperparams = self.hps.get("DepPairingLayer", {}),
                                    input_size = self.word_lstm.output_size,
                                    output_size = self.task_dims["link_label"],
                                    )

        # nt = 3 if tree_bidirectional else 1
        # ns = 2 if lstm_bidirectional else 1
        # tree_input_size = seq_lstm_h_size * ns + \
        #     dep_embs_size + self.num_seg_labels
        # link_label_input_size = tree_lstm_h_size * nt + 2 * seq_lstm_h_size * ns

        # self.link_label_output_size = 2 * (num_link_labels - 1) + 1
        # self.link_label_clf = DepPairingLayer(
        #                                     tree_input_size=tree_input_size,
        #                                     tree_lstm_h_size=tree_lstm_h_size,
        #                                     tree_bidirectional=tree_bidirectional,
        #                                     decoder_input_size=link_label_input_size,
        #                                     decoder_h_size=link_label_clf_h_size,
        #                                     decoder_output_size=self.link_label_output_size,
        #                                     dropout=dropout)


    @classmethod
    def name(self):
        return "LSTM_ER"

    def forward(self, batch):

        token_mask = batch["token"]["mask"].type(torch.bool)

        # pos_word_embs.shape =
        # (batch_size, max_nr_tokens, word_embs + pos_embs)
        pos_word_embs = torch.cat((batch["token"]["word_embs"], batch["token"]["pos_embs"]), dim=2)
        pos_word_embs = self.dropout(pos_word_embs)

        # lstm_out = (batch_size, max_nr_tokens, lstm_hidden)
        lstm_out, _ = self.lstm(pos_word_embs, batch["token"]["lengths"])

        # outpts the logits, preds for seg+label as well as segmentation data which contain
        # information about where segments start, in this cases it decoded BIO patterns
        sl_logits, sl_preds, seg_data = self.segmenter(
                                                        input = lstm_out,
                                                        lengths = batch["token"]["lengths"],
                                                        batch = batch
                                                        )

        one_hot_embs = utils.one_hot(
                                    preds = sl_preds, 
                                    mask = token_mask,
                                    nr_labels = self.num_seg_labels
                                    )

        ll_logits, ll_preds, l_preds, pair_data  = self.link_labeler(
                                                                    token_embs = lstm_out,
                                                                    dep_embs = batch["token"]["deprel_embs"],
                                                                    one_hot_embs = one_hot_embs,
                                                                    roots = batch["token"]["root_idxs"],
                                                                    deplinks = batch["token"]["dephead"],
                                                                    token_mask = token_mask,
                                                                    seg_data = seg_data,
                                                                    )


        return {  
                "logits": {
                            "seg+label": sl_logits,
                            "link_label": ll_logits,
                            },
                "preds":
                        {
                            "seg+label": sl_preds,
                            "link_label": ll_preds,
                            "link": l_preds
                        },
                "pair_data": pair_data

                 }



    def loss(self, batch, forward_outputs:dict):

        seg_label_loss = self.segmenter.loss(
                                            targets =  batch["token"]["seg+label"],
                                            logits = forward_outputs["logits"]["seg+label"],
                                            )

        link_label_loss = self.link_labeler.loss(
                                                targets = batch["token"]["link_label"],
                                                logits = forward_outputs["logits"]["link_label"],
                                                token_preds = forward_outputs["preds"]["seg+label"],
                                                token_targets = batch["token"]["seg+label"],
                                                pair_data = forward_outputs["pair_data"]
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