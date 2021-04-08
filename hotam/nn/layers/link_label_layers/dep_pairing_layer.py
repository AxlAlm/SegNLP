from typing import List, Tuple, DefaultDict

import functools

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
from networkx import Graph as nxGraph
import dgl
from dgl import DGLGraph
from dgl.traversal import topological_nodes_generator as traverse_topo

from hotam.nn.layers.type_treelstm import TypeTreeLSTM
from hotam.nn.utils import index_4D, unfold_matrix


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

        self.tree_lstm_h_size = tree_lstm_h_size
        self.tree_lstm_bidir = tree_bidirectional

        self.tree_lstm = TypeTreeLSTM(embedding_dim=tree_input_size,
                                      h_size=tree_lstm_h_size,
                                      dropout=dropout,
                                      bidirectional=tree_bidirectional)

        self.label_link_clf = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_h_size), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(decoder_h_size,
                                           decoder_output_size))

        self.__supported_modes = set(["shortest_path"])

    def assert_graph(self, u: Tensor, v: Tensor, roots: Tensor,
                     self_loop: Tensor) -> None:

        # check that self loops do not exist in places other than roots
        self_loop_check = u == roots[:, None]
        if not torch.all(self_loop == self_loop_check):
            # we have a problem. Get samples ids where we have more than one
            # self loop
            problem_id = torch.where(torch.sum(self_loop, 1) > 1)[0]
            self_loop_id = v[problem_id, :][self_loop[problem_id, :]].tolist()
            theroot = roots[problem_id].tolist()
            error_msg_1 = f"Self loop found in sample(s): {problem_id}, "
            error_msg_2 = f"Node(s) {self_loop_id}. Root(s): {theroot}."
            raise Exception(error_msg_1 + error_msg_2)
            # remove the sample that has the problem

    def get_sample_graph(self, deplinks: Tensor, roots: Tensor,
                         token_mask: Tensor, assertion: bool) -> List[Tensor]:

        batch_size = deplinks.size(0)
        max_lenght = deplinks.size(1)
        device = torch.device("cpu")

        # G(U, V)
        U = torch.arange(max_lenght, device=self.device).repeat(batch_size, 1)
        V = deplinks.clone()
        M = token_mask.clone()

        # remove self loops at root nodes
        self_loop = U == V
        if assertion:
            self.assert_graph(U, V, roots, self_loop)
        U = U[~self_loop].view(batch_size, -1).to(device)
        V = V[~self_loop].view(batch_size, -1).to(device)
        M = M[~self_loop].view(batch_size, -1).to(device)

        return [U, V, M]

    def get_subgraph(self,
                     start: int,
                     end: int,
                     g: DGLGraph,
                     g_nx: nxGraph,
                     sub_graph_type: str,
                     assertion: bool = False) -> DGLGraph:
        """
        """
        if sub_graph_type == "shortest_path":
            thepath = nx.shortest_path(g_nx, source=start, target=end)
            sub_g = dgl.node_subgraph(g, thepath)
            root = list(traverse_topo(sub_g))[-1]

            # initialize node data
            # Node type
            node_type = torch.zeros(sub_g.number_of_nodes(), dtype=torch.long)
            sub_g.ndata["type_n"] = node_type

            # Root, start and end leaves node
            str_mark = torch.zeros(sub_g.number_of_nodes())
            end_mark = torch.zeros_like(str_mark)
            root_mark = torch.zeros_like(str_mark)
            str_mark[0] = 1
            end_mark[-1] = 1
            root_mark[root] = 1
            sub_g.ndata["root"] = root_mark
            sub_g.ndata["start"] = str_mark
            sub_g.ndata["end"] = end_mark

            # check ...
            if assertion:
                assert len(root) == 1
                assert sub_g.ndata["_ID"][0] == start
                assert sub_g.ndata["_ID"][-1] == end
                assert str_mark.sum() == end_mark.sum()
                assert str_mark.sum() == root_mark.sum()

        elif sub_graph_type == 1:
            # get subtree
            pass
        return sub_g

    def build_dep_graphs(self, token_embs: Tensor, deplinks: Tensor,
                         roots: Tensor, token_mask: Tensor, token_reps: Tensor,
                         subgraphs: List[List[Tuple]], mode: str,
                         assertion: bool) -> List[DGLGraph]:

        batch_size = deplinks.size(0)

        # creat sample graphs G(U, V) tensor on CPU
        U, V, M = self.get_sample_graph(deplinks=deplinks,
                                        roots=roots,
                                        token_mask=token_mask,
                                        assertion=assertion)
        # creat sub_graph for each pair
        dep_graphs = []
        nodes_emb = []
        for b_i in range(batch_size):
            mask = M[b_i]
            u = U[b_i][mask]
            v = V[b_i][mask]
            # creat sample DGLGraph, convert it to unidirection, separate the
            # list of tuples candidate pairs into two lists: (start and end
            # tokens), then create subgraph
            graph = dgl.graph((u, v))
            graph_unidir = graph.to_networkx().to_undirected()
            start, end = list(zip(*subgraphs[b_i]))
            if start == []:  # no candidate pair
                continue

            subgraph_func = functools.partial(self.get_subgraph,
                                              g=graph,
                                              g_nx=graph_unidir,
                                              sub_graph_type=mode,
                                              assertion=assertion)
            dep_graphs.append(dgl.batch(list(map(subgraph_func, start, end))))
            # get nodes' token embedding
            nodes_emb.append(token_embs[b_i, dep_graphs[b_i].ndata["_ID"]])

        # batch graphs, move to model device, update nodes data by tokens
        # embedding
        nodes_emb = torch.cat(nodes_emb, dim=0)
        dep_graphs = dgl.batch(dep_graphs).to(self.device)
        dep_graphs.ndata["emb"] = nodes_emb

        return dep_graphs

    def forward(self,
                input_embs: Tensor,
                unit_repr: Tensor,
                unit_num: list,
                dependencies: Tensor,
                token_mask: Tensor,
                roots: Tensor,
                pairs: DefaultDict[str, List[List[Tuple[int]]]],
                mode: str = "shortest_path",
                assertion: bool = False):

        mode_bool = mode in self.__supported_modes
        assert mode_bool, f"{mode} is not a supported mode for DepPairingLayer"

        self.device = input_embs.device
        dir_n = 2 if self.tree_lstm_bidir else 1
        batch_size = input_embs.size(0)

        # 8)
        dep_graphs = self.build_dep_graphs(token_embs=input_embs,
                                           deplinks=dependencies,
                                           roots=roots,
                                           token_mask=token_mask,
                                           token_reps=input_embs,
                                           subgraphs=pairs["end"],
                                           mode=mode,
                                           assertion=assertion)

        # 9)
        h0 = torch.zeros(dep_graphs.num_nodes(),
                         self.tree_lstm_h_size,
                         device=self.device)
        c0 = torch.zeros_like(h0)
        tree_lstm_out = self.tree_lstm(dep_graphs, h0, c0)

        # construct dp = [↑hpA; ↓hp1; ↓hp2]
        # ↑hpA: hidden state of dep_graphs' root
        # ↓hp1: hidden state of the first token in the candidate pair
        # ↓hp2: hidden state of the second token in the candidate pair
        # get ids of roots and tokens in relation
        root_id = (dep_graphs.ndata["root"] == 1)
        start_id = dep_graphs.ndata["start"] == 1
        end_id = dep_graphs.ndata["end"] == 1
        tree_lstm_out = tree_lstm_out.view(-1, dir_n, self.tree_lstm_h_size)
        tree_logits = tree_lstm_out[root_id, 0, :]  # ↑hpA
        if self.tree_lstm_bidir:
            hp1 = tree_lstm_out[start_id, 1, :]  # ↓hp1
            hp2 = tree_lstm_out[end_id, 1, :]  # ↓hp2
            tree_logits = torch.cat((tree_logits, hp1, hp2), dim=-1)
        # [dp; s]
        link_label_input_repr = torch.cat((tree_logits, unit_repr), dim=-1)
        logits = self.label_link_clf(link_label_input_repr)
        prob_dist = F.softmax(logits, dim=-1)

        # NOTE: When there are multiple equal values, torch.argmax() and
        # torch.max() do not return the first max value.  Instead, they
        # randamly return any valid index.

        # reshape logits and prob into 4d tensors
        size_4d = [
            batch_size,
            max(unit_num),
            max(unit_num),
            prob_dist.size(-1)
        ]
        prob_4d = input_embs.new_ones(size=size_4d) * -1
        logits_4d = input_embs.new_ones(size=size_4d) * -1
        unit_start = []
        unit_end = []

        # split prob_dist and logits based on number of pairs in each batch
        pair_num = list(map(len, pairs["end"]))
        prob_ = torch.split(prob_dist, split_size_or_sections=pair_num)
        logits_ = torch.split(logits, split_size_or_sections=pair_num)

        # fill 4d tensors
        data = zip(prob_, logits_, pairs["start"], pairs["end"], unit_num)
        for i, (prob, logs, s, e, n) in enumerate(data):
            if n > 0:
                prob_4d[i, :n, :n] = prob.view(n, n, -1)
                logits_4d[i, :n, :n] = logs.view(n, n, -1)
                unit_start.extend(list(zip(*s))[1][:n])
                unit_end.extend(list(zip(*e))[1][:n])

        # get link label max logits and link_label and link predictions
        prob_max_pair, _ = torch.max(prob_4d, dim=-1)
        link_preds = torch.argmax(prob_max_pair, dim=-1)
        link_label_prob_dist = index_4D(prob_4d, index=link_preds)
        link_label_preds = torch.argmax(link_label_prob_dist, dim=-1)
        link_label_max_logits = index_4D(logits_4d, index=link_preds)

        link_label_max_logits = unfold_matrix(
            matrix_to_fold=link_label_max_logits,
            start_idx=unit_start,
            end_idx=unit_end,
            class_num_betch=unit_num,
            fold_dim=input_embs.size(1))

        link_label_preds = unfold_matrix(matrix_to_fold=link_label_preds,
                                         start_idx=unit_start,
                                         end_idx=unit_end,
                                         class_num_betch=unit_num,
                                         fold_dim=input_embs.size(1))

        link_preds = unfold_matrix(matrix_to_fold=link_preds,
                                   start_idx=unit_start,
                                   end_idx=unit_end,
                                   class_num_betch=unit_num,
                                   fold_dim=input_embs.size(1))

        # Would not be easier to save a mapping ac_id, token_strat/end, instead
        # of all of these calc and loops !
        return [link_label_max_logits, link_label_preds, link_preds]
