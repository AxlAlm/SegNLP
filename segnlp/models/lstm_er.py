      
      
      
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


    @classmethod
    def name(self):
        return "LSTM_ER"

    
    def encoding():
        # (batch_size, max_nr_tokens, word_embs + pos_embs)
        pos_word_embs = torch.cat((batch["token"]["word_embs"], batch["token"]["pos_embs"]), dim=2)
        pos_word_embs = self.dropout(pos_word_embs)

        # lstm_out = (batch_size, max_nr_tokens, lstm_hidden)
        lstm_out, _ = self.lstm(pos_word_embs, batch["token"]["lengths"])



    def forward(self, batch, output):

        # (batch_size, max_nr_tokens, word_embs + pos_embs)
        pos_word_embs = torch.cat((batch["token"]["word_embs"], batch["token"]["pos_embs"]), dim=2)
        pos_word_embs = self.dropout(pos_word_embs)

        # lstm_out = (batch_size, max_nr_tokens, lstm_hidden)
        lstm_out, _ = self.lstm(pos_word_embs, batch["token"]["lengths"])

        # outpts the logits, preds for seg+label as well as segmentation data which contain
        # information about where segments start, in this cases it decoded BIO patterns
        output.add(self.segmenter(
                        input = lstm_out,
                        lengths = batch["token"]["lengths"],
                        ))


        output.add(self.link_labeler(
                        token_embs = lstm_out,
                        dep_embs = batch["token"]["deprel_embs"],
                        one_hot_embs = output.one_hots(TASK),
                        roots = batch["token"]["root_idxs"],
                        deplinks = batch["token"]["dephead"],
                        token_mask = batch["token"]["mask"].type(torch.bool),
                        ))

        return output




    def loss(self, batch, output):

        seg_label_loss = self.segmenter.loss(
                                            targets =  batch["token"]["seg+label"],
                                            logits = output.logits["seg+label"],
                                            )

        link_label_loss = self.link_labeler.loss(
                                                targets = batch["token"]["link_label"],
                                                logits = output.logits["link_label"],
                                                token_preds = output.logits["seg+label"],
                                                token_targets = batch.logits["seg+label"],
                                                pair_data = output["pair_data"]
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