

# basics
from typing import List, Tuple, DefaultDict, Dict
import functools
from itertools import chain, repeat
# import time

import numpy as np
import pandas as pd

# pytorch
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# networkx
import networkx as nx
from networkx import Graph as nxGraph

# DGL
import dgl
from dgl import DGLGraph
from dgl.traversal import topological_nodes_generator as traverse_topo
from torch.nn.modules.loss import CrossEntropyLoss

# hotam
from segnlp.nn.layers.rep_layers import TypeTreeLSTM

from segnlp.utils import get_all_possible_pairs
from segnlp.utils import range_3d_tensor_index

class DepGraph:
    def __init__(self,
                 token_embs: Tensor,
                 deplinks: Tensor,
                 roots: Tensor,
                 token_mask: Tensor,
                 subgraphs: List[Tensor],
                 mode: str,
                 device=None,
                 assertion: bool = False) -> List[DGLGraph]:

        assert mode in set([
            "shortest_path"
        ]), f"{mode} is not a supported mode for DepPairingLayer"
        self.device = device
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
            if not subgraphs[b_i]:  # no candidate pair
                continue
            mask = M[b_i]
            u = U[b_i][mask]
            v = V[b_i][mask]
            # creat sample DGLGraph, convert it to unidirection, separate the
            # list of tuples candidate pairs into two lists: (start and end
            # tokens), then create subgraph
            graph = dgl.graph((u, v))
            graph_unidir = graph.to_networkx().to_undirected()
            start, end = list(zip(*subgraphs[b_i]))

            subgraph_func = functools.partial(self.get_subgraph,
                                              g=graph,
                                              g_nx=graph_unidir,
                                              sub_graph_type=mode,
                                              assertion=assertion)

            dep_graphs.append(dgl.batch(list(map(subgraph_func, start, end))))

            # get nodes' token embedding
            nodes_emb.append(token_embs[b_i, dep_graphs[-1].ndata["_ID"]])

        # batch graphs, move to model device, update nodes data by tokens
        # embedding
        nodes_emb = torch.cat(nodes_emb, dim=0)
        self.graphs = dgl.batch(dep_graphs).to(self.device)
        self.graphs.ndata["emb"] = nodes_emb

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






# basics
from typing import List, Tuple, DefaultDict, Dict
import functools
from itertools import chain, repeat
# import time

import numpy as np
import pandas as pd

# pytorch
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# networkx
import networkx as nx
from networkx import Graph as nxGraph

# DGL
import dgl
from dgl import DGLGraph
from dgl.traversal import topological_nodes_generator as traverse_topo
from torch.nn.modules.loss import CrossEntropyLoss

# hotam
from segnlp.nn.layers.rep_layers import TypeTreeLSTM

from segnlp.utils import get_all_possible_pairs
from segnlp.utils import range_3d_tensor_index

class DepGraph:
    def __init__(self,
                 token_embs: Tensor,
                 deplinks: Tensor,
                 roots: Tensor,
                 token_mask: Tensor,
                 subgraphs: List[Tensor],
                 mode: str,
                 device=None,
                 assertion: bool = False) -> List[DGLGraph]:

        assert mode in set([
            "shortest_path"
        ]), f"{mode} is not a supported mode for DepPairingLayer"
        self.device = device
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
            if not subgraphs[b_i]:  # no candidate pair
                continue
            mask = M[b_i]
            u = U[b_i][mask]
            v = V[b_i][mask]
            # creat sample DGLGraph, convert it to unidirection, separate the
            # list of tuples candidate pairs into two lists: (start and end
            # tokens), then create subgraph
            graph = dgl.graph((u, v))
            graph_unidir = graph.to_networkx().to_undirected()
            start, end = list(zip(*subgraphs[b_i]))

            subgraph_func = functools.partial(self.get_subgraph,
                                              g=graph,
                                              g_nx=graph_unidir,
                                              sub_graph_type=mode,
                                              assertion=assertion)

            dep_graphs.append(dgl.batch(list(map(subgraph_func, start, end))))

            # get nodes' token embedding
            nodes_emb.append(token_embs[b_i, dep_graphs[-1].ndata["_ID"]])

        # batch graphs, move to model device, update nodes data by tokens
        # embedding
        nodes_emb = torch.cat(nodes_emb, dim=0)
        self.graphs = dgl.batch(dep_graphs).to(self.device)
        self.graphs.ndata["emb"] = nodes_emb

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


class DepTreeLSTM(nn.Module):

    def __init__(self,
                 tree_input_size: int,
                 tree_lstm_h_size: int,
                 tree_bidirectional: bool,
                 decoder_input_size: int,
                 decoder_h_size: int,
                 decoder_output_size: int,
                 mode : str = "shortest_path",
                 dropout: int = 0.0
                 ):
        super(self).__init__()

        self.mode = "shortest_path"

        self.tree_lstm_h_size = tree_lstm_h_size
        self.tree_lstm_bidir = tree_bidirectional

        self.tree_lstm = TypeTreeLSTM(embedding_dim=tree_input_size,
                                      h_size=tree_lstm_h_size,
                                      dropout=dropout,
                                      bidirectional=True)

        self.link_label_clf_layer = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_h_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(decoder_h_size, decoder_output_size),
        )
        # number of link labels including None
        self.ll_num = (decoder_output_size + 1) // 2


    def split_nested_list(self,
                          nested_list: List[list],
                          device=torch.device) -> Tuple[Tensor]:

        list_1, list_2 = list(zip(*chain.from_iterable(nested_list)))
        list_1 = torch.tensor(list_1, dtype=torch.long, device=device)
        list_2 = torch.tensor(list_2, dtype=torch.long, device=device)
        return list_1, list_2


    def forward(self,
                        token_embs: Tensor,
                        dep_embs: Tensor,
                        one_hot_embs: Tensor,
                        roots: Tensor,
                        token_mask: Tensor,
                        deplinks: Tensor,
                        pair_data: dict,
                        assertion: bool = False
                        ):

        # 1) Build graph from dependecy data
        node_embs = torch.cat((token_embs, one_hot_embs, dep_embs), dim=-1)

        G = DepGraph(
                    token_embs=node_embs,
                    deplinks=deplinks,
                    roots=roots,
                    token_mask=token_mask,
                    subgraphs=pair_data["end"],
                    mode=self.mode,
                    device=self.device,
                    assertion=assertion
                    )

        # 2) Pass graph to a TreeLSTM to create hidden representations
        h0 = torch.zeros(G.graphs.num_nodes(),
                         self.tree_lstm_h_size,
                         device=self.device)
        c0 = torch.zeros_like(h0)
        tree_lstm_out = self.tree_lstm(G.graphs, h0, c0)

        # 3) construct dp = [↑hpA; ↓hp1; ↓hp2]
        # ↑hpA: hidden state of dep_graphs' root
        # ↓hp1: hidden state of the first token in the candidate pair
        # ↓hp2: hidden state of the second token in the candidate pair
        # get ids of roots and tokens in relation
        root_id = G.graphs.ndata["root"] == 1
        start_id = G.graphs.ndata["start"] == 1
        end_id = G.graphs.ndata["end"] == 1

        tree_lstm_out = tree_lstm_out.view(-1, 2, self.tree_lstm_h_size)
        tree_pair_embs = torch.cat(
                    (
                        tree_lstm_out[root_id, 0, :],  # ↑hpA
                        tree_lstm_out[start_id, 1, :],  # ↓hp1
                        tree_lstm_out[end_id, 1, :]  # ↓hp2
                    ),
                    dim=-1)

        return tree_pair_embs



