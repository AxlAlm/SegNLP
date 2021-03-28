from math import floor
from random import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from hotam.nn.layer.seg_layers import BigramSegLayer
from hotam.nn.layers.link_label_layers import DepPairingLayer
from hotam.nn.layers.lstm import LSTM_LAYER
from hotam.nn.layers.type_treelstm import TypeTreeLSTM
from hotam.nn.utils import index_4D, get_all_possible_pairs
from hotam.nn.schedule_sample import ScheduleSampling


class LSTM_ER(nn.Module):
    def __init__(self, hyperparamaters: dict, task2dim: dict,
                 feature2dim: dict):
        super(LSTM_ER, self).__init__()

        # number of arguemnt components
        self.num_ac = len(task2dim["seg_ac"])
        self.num_relations = len(task2dim["stance"])  # number of relations
        self.no_stance = task2dim["stance"].index("None")

        self.model_param = nn.Parameter(th.empty(0))

        self.sub_graph_type = hyperparamaters["sub_graph_type"]

        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.BATCH_SIZE = hyperparamaters["batch_size"]

        token_embs_size = feature2dim["word_embs"] + feature2dim["pos_embs"]
        label_embs_size = self.num_ac
        dep_embs_size = feature2dim["deprel_embs"]

        seq_lstm_h_size = hyperparamaters["seq_lstm_h_size"]
        self.tree_lstm_h_size = hyperparamaters["tree_lstm_h_size"]

        clf_h_size = hyperparamaters["ac_seg_hidden_size"]
        clf_output_size = self.num_ac

        re_hidden_size = hyperparamaters["re_hidden_size"]
        re_output_size = self.num_relations

        seq_lstm_num_layers = hyperparamaters["seq_lstm_num_layers"]

        lstm_bidirectional = hyperparamaters["lstm_bidirectional"]
        tree_bidirectional = hyperparamaters["tree_bidirectional"]

        dropout = hyperparamaters["dropout"]
        self.dropout = nn.Dropout(dropout)

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
        self.schedule = ScheduleSampling(schedule="exponential",
                                         k=hyperparamaters["k"])

        self.lstm = LSTM_LAYER(input_size=token_embs_size,
                               hidden_size=seq_lstm_h_size,
                               num_layers=seq_lstm_num_layers,
                               bidirectional=lstm_bidirectional,
                               dropout=0.0)

        num_dirs = 2 if lstm_bidirectional else 1
        clf_input_size = label_embs_size + (seq_lstm_h_size * num_dirs)
        self.seg_label_clf = BigramSegLayer(
            input_size=clf_input_size,
            hidden_size=clf_h_size,
            output_size=clf_output_size,
            label_emb_dim=label_embs_size,
            dropout=dropout,
        )

        self.link_label_clf = self.DepPairingLayer()

        nt = 3 if tree_bidirectional else 1
        ns = 2 if lstm_bidirectional else 1
        re_input_size = self.tree_lstm_h_size * nt + 2 * seq_lstm_h_size * ns
        tree_input_size = seq_lstm_h_size * ns + dep_embs_size + label_embs_size
        self.tree_lstm = TypeTreeLSTM(embedding_dim=tree_input_size,
                                      h_size=self.tree_lstm_h_size,
                                      dropout=dropout,
                                      bidirectional=tree_bidirectional)
        self.rel_decoder = nn.Sequential(
            nn.Linear(re_input_size, re_hidden_size), nn.Tanh(), self.dropout,
            nn.Linear(re_hidden_size, re_output_size))

    def forward(self, batch, output):

        # 1)
        # pos_word_embs.shape =
        # (batch_size, max_nr_tokens, word_embs + pos_embs)
        pos_word_embs = th.cat(
            (batch["token"]["word_embs"], batch["token"]["pos_embs"]), dim=2)
        pos_word_embs = self.dropout(pos_word_embs)

        # 2) lstm_out = (batch_size, max_nr_tokens, lstm_hidden)
        lstm_out, _ = self.lstm(pos_word_embs, batch["token"]["lengths"])

        # 3)
        # seg_label_logits = (batch_size, max_nr_tokens, nr_labels)
        # seg_label_probs = (batch_size, max_nr_tokens, nr_labels)
        # seg_label_preds = (batch_size, max_nr_tokens)
        # one_hots = (batch_size, max_nr_tokens, nr_layers)
        clf_output = self.seg_label_clf(lstm_out, batch["token"]["lengths"])
        seg_label_logits, seg_label_probs = clf_output[:2]
        seg_label_preds, seg_label_embs = clf_output[-2:]

        # 4)
        coin_flip = floor(random() * 10) / 10
        if self.schedule.next() >= coin_flip:
            preds_used = batch["token"]["seg+label"]
            embs_used = F.one_hots(preds_used, num_classes=self.num_ac)
        else:
            embs_used = seg_label_embs

        # 5)
        span_lengths, none_span_mask, nr_units = bio_decode(
            batch_encoded_bios=preds,
            lengths=batch["token"]["lengths"],
            apply_correction=True,
            B=[],  #ids for labels counted as B
            I=[],  #ids for labels counted as I
            O=[],  #ids for labels counted as O
        )

        #6)
        #NOTE! we can change this to output a tensor or array if its suits better.
        all_possible_pairs = get_all_possible_pairs(span_lengths,
                                                    none_unit_mask)

        #7)
        node_embs = th.cat((lstm_out, one_hots, batch["token"]["dep_embs"]),
                           dim=-1)

        #8)
        link_label_logits, link_preds = self.link_label_clf(
            input_embs=node_embs,
            dependecies=batch["token"]["dephead"],
            pairs=all_possible_pairs,
            mode="shortest_path")

        if self.train_mode:

            #CALCULATE LOSS HERE

            output.add_loss(task="total", data=total_loss)
            output.add_loss(task="link_label", data=link_label_loss)
            output.add_loss(task="label", data=label_loss)

        output.add_preds(task="seg+label", level="token", data=label_preds)
        output.add_preds(task="link", level="unit", data=link_preds)
        output.add_preds(task="link_label",
                         level="unit",
                         data=link_label_preds)

        return output
