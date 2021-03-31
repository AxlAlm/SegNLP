from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn as nn

import dgl

from hotam.nn.layers.type_treelstm import TypeTreeLSTM
from hotam.nn.utils import index_4D


class DepPairingLayer(nn.Module):
    def __init__(self,
                 tree_input_size: int,
                 tree_lstm_h_size: int,
                 tree_bidirectional: bool,
                 decoder_input_size: int,
                 decoder_h_size: int,
                 decoder_output_size: int,
                 dropout: int = 0.0):
        super(DepPairingLayer, self).__init__()

        self.tree_lstm = TypeTreeLSTM(embedding_dim=tree_input_size,
                                      h_size=tree_lstm_h_size,
                                      dropout=dropout,
                                      bidirectional=tree_bidirectional)

        self.label_link_clf = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_h_size), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(decoder_h_size,
                                           decoder_output_size))

        self.__supported_modes = set(["shortest_path"])

    def assert_graph(self,
                     u: Tensor,
                     v: Tensor,
                     roots: Tensor,
                     self_loop: Tensor,
                     mode: str = "self looping") -> None:
        # check that self loops do not exist in places other than roots
        self_loop_check = u == roots[:, None]
        if not torch.all(self_loop == self_loop_check):
            # we have a problem. Get samples ids where we have more than one
            # self loop
            problem_id = torch.where(torch.sum(self_loop, 1) > 1)[0].tolist()
            self_loop_id = v[problem_id, :][self_loop[problem_id, :]].tolist()
            theroot = roots[problem_id].tolist()
            error_msg_1 = f"Self loop found in sample(s): {problem_id}, "
            error_msg_2 = f"Node(s) {self_loop_id}. Root(s): {theroot}."
            raise Exception(error_msg_1 + error_msg_2)

    def build_dep_graphs(self, deplinks: Tensor, roots: Tensor,
                         token_mask: Tensor, token_reps: Tensor,
                         subgraphs: List[List[Tuple]], mode: str):
        # Craete graph G(u,v)
        batch_size = deplinks.size(0)
        max_lenght = deplinks.size(1)
        device = deplinks.device

        U = torch.arange(max_lenght).repeat(batch_size, 1)
        # remove self loops at root nodes
        self_loop = U == deplinks
        self.assert_graph(U, deplinks, roots, self_loop)

    def forward(self,
                input_embs: Tensor,
                dependencies: Tensor,
                token_mask: Tensor,
                roots: Tensor,
                pairs: List[List[Tuple]],
                mode: str = "shortest_path"):

        mode_bool = mode in self.__supported_modes
        assert mode_bool, f"{mode} is not a supported mode for DepPairingLayer"

        # 8)

        dep_graphs = self.build_dep_graphs(deplinks=dependencies,
                                           roots=roots,
                                           token_mask=token_mask,
                                           token_reps=input_embs,
                                           subgraphs=pairs,
                                           mode=mode)

        # 9)
        #
        # tree_lstm_out = self.tree_lstm(graphs)

        # 10) Here we should format the data to the following structure:
        # t1 = representation of the last token in the first unit of the pair
        # t2 = representation of the last token in the second unit of the pair
        # a = lowest ancestor of t1 and t2
        # pair(unit_i, unit_j) = a+t1+t2 where t1
        # (batch_size, nr_units, nr_units, a+t1+t2)
        # for a sample:
        # [
        #   [
        #    pair(unit0,unit0),
        #       ...
        #     pair(unit0, unitn),
        #   ],
        #    ....
        #   [
        #    pair(unitn,unitn),
        #       ...
        #     pair(unitn, unitn+1),
        #   ],
        #
        # ]
        # pairs = ""

        # now we should get logist  for each link_labels
        # (batch_size, nr_units, nr_units, nr_link_labels)
        #
        # for a sample:
        # [
        #   [
        #    [link_label_0_score, .., link_label_n_score],
        #       ...
        #    [link_label_0_score, .., link_label_n_score]
        #   ],
        #   [
        #    [link_label_0_score, .., link_label_n_score],
        #       ....
        #    [link_label_0_score, .., link_label_n_score]
        #   ],
        # ]
        # link_label_logits = self.link_label_clf(pairs)

        # 11)
        # first we get the index of the unit each unit links to
        # we do this by first get the highest score of the link label
        # for each unit pair. Then we argmax that to get the index of
        # the linked unit.
        # max_link_label_logits = torch.max(link_label_logits, dim=-1)
        # link_preds = torch.argmax(max_link_label_logits, dim=-1)

        # 12)
        # we index the link_label_scores by the link predictions, selecting
        # the logits for the link_labels for the linked pairs
        # top_link_label_logits = index_4D(link_label_logits, index=link_preds)

        # return top_link_label_logits, link_preds
