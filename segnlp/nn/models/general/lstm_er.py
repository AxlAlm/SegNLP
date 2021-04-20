from functools import reduce
from operator import iconcat

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from segnlp.nn.layers.seg_layers.bigram_seg import BigramSegLayer
from segnlp.nn.layers.link_label_layers import DepPairingLayer
from segnlp.nn.layers.rep_layers import LSTM
from segnlp.nn.utils import get_all_possible_pairs
#from segnlp.nn.utils import range_3d_tensor_index
from segnlp.nn.utils import ScheduleSampling
from segnlp.nn.utils import bio_decode


class LSTM_ER(nn.Module):
    def __init__(self, hyperparamaters:dict, task_dims:dict, feature_dims:dict, inference:bool):
        super().__init__()
        self.inference = inference

        # number of arguemnt components
        self.num_ac = task_dims["seg+label"]
        self.num_stances = task_dims["link_label"]  # number of relations

        self.model_param = nn.Parameter(torch.empty(0))

        self.sub_graph_type = hyperparamaters["sub_graph_type"]

        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.BATCH_SIZE = hyperparamaters["batch_size"]

        token_embs_size = feature_dims["word_embs"] + feature_dims["pos_embs"]
        label_embs_size = self.num_ac
        dep_embs_size = feature_dims["deprel_embs"]

        seq_lstm_h_size = hyperparamaters["seq_lstm_h_size"]
        tree_lstm_h_size = hyperparamaters["tree_lstm_h_size"]

        clf_h_size = hyperparamaters["ac_seg_hidden_size"]
        clf_output_size = self.num_ac

        link_label_clf_h_size = hyperparamaters["re_hidden_size"]

        seq_lstm_num_layers = hyperparamaters["seq_lstm_num_layers"]

        lstm_bidirectional = hyperparamaters["lstm_bidirectional"]
        tree_bidirectional = hyperparamaters["tree_bidirectional"]

        dropout = hyperparamaters["dropout"]
        self.dropout = nn.Dropout(dropout)

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
        self.schedule = ScheduleSampling(schedule="inverse_sig",
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

        nt = 3 if tree_bidirectional else 1
        ns = 2 if lstm_bidirectional else 1
        tree_input_size = seq_lstm_h_size * ns + dep_embs_size + label_embs_size
        link_label_input_size = tree_lstm_h_size * nt + 2 * seq_lstm_h_size * ns
        self.link_label_clf = DepPairingLayer(
            tree_input_size=tree_input_size,
            tree_lstm_h_size=tree_lstm_h_size,
            tree_bidirectional=tree_bidirectional,
            decoder_input_size=link_label_input_size,
            decoder_h_size=link_label_clf_h_size,
            decoder_output_size=self.num_stances,
            dropout=dropout)


    @classmethod
    def name(self):
        return "LSTM_ER"


    def forward(self, batch, output):

        check = True
        # 1)
        # pos_word_embs.shape =
        # (batch_size, max_nr_tokens, word_embs + pos_embs)
        pos_word_embs = torch.cat(
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

        if self.schedule.next(batch.current_epoch):
            preds_used = batch["token"]["seg+label"]
            embs_used = F.one_hot(preds_used, num_classes=self.num_ac)
        else:
            preds_used = seg_label_preds
            embs_used = seg_label_embs

        # 5)
        # NOTE It would be better to have during initialization instead of
        # getting the same labels ids each step
        bio_dict = defaultdict(list)
        for (i, label) in enumerate(output.label_encoders["seg+label"].labels):
            bio_dict[label[0]].append(i)
        span_lengths, none_unit_mask, nr_units = bio_decode(
            batch_encoded_bios=preds_used,
            lengths=batch["token"]["lengths"],
            apply_correction=True,
            B=bio_dict["B"],  # ids for labels counted as B
            I=bio_dict["I"],  # ids for labels counted as I
            O=bio_dict["O"],  # ids for labels counted as O
        )

        # 6)
        all_possible_pairs = get_all_possible_pairs(span_lengths,
                                                    none_unit_mask,
                                                    assertion=check)

        # 7)
        node_embs = torch.cat(
            (lstm_out, embs_used, batch['token']['deprel_embs']), dim=-1)
        # construct Si: average of sequential lstm hidden state for each unit
        # flatten List[List[Tuple[int]]] to List[Tuple[int]], separate the
        # list of tuples candidate pairs into two lists
        units_start_ids = reduce(iconcat, all_possible_pairs["start"], [])
        units_end_ids = reduce(iconcat, all_possible_pairs["end"], [])
        unit1_start, unit2_start = np.array(list(zip(*units_start_ids)))
        unit1_end, unit2_end = np.array(list(zip(*units_end_ids)))
        # number of pairs in each sample
        units_pair_num = list(map(len, all_possible_pairs["end"]))

        # Indexing sequential lstm hidden state for each unit
        unit1_s = range_3d_tensor_index(lstm_out,
                                        unit1_start,
                                        unit1_end,
                                        units_pair_num,
                                        reduce_="mean")
        unit2_s = range_3d_tensor_index(lstm_out,
                                        unit2_start,
                                        unit2_end,
                                        units_pair_num,
                                        reduce_="mean")
        s = torch.cat((unit1_s, unit2_s), dim=-1)

        # 8)
        link_label_logits, link_preds = self.link_label_clf(
            input_embs=node_embs,
            unit_repr=s,
            dependencies=batch["token"]["dephead"],
            token_mask=batch["token"]["mask"],
            roots=output.batch["token"]["root_idxs"],
            pairs=all_possible_pairs["end"],
            mode="shortest_path",
            assertion=check)

        # if self.train_mode:
        #     #CALCULATE LOSS HERE

        #     output.add_loss(task="total", data=total_loss)
        #     output.add_loss(task="link_label", data=link_label_loss)
        #     output.add_loss(task="label", data=label_loss)

        # output.add_preds(task="seg+label", level="token", data=label_preds)
        # output.add_preds(task="link", level="unit", data=link_preds)
        # output.add_preds(task="link_label",
        #                  level="unit",
        #                  data=link_label_preds)

        return 1
