

# basics
from typing import List, Sequence, Tuple, DefaultDict, Dict, Union
import functools
from itertools import chain, repeat
import numpy as np
from numpy.lib.arraysetops import isin
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


#segnlp
from segnlp import utils


class TreeLSTMCell(nn.Module):
    def __init__(self, xemb_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(xemb_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))

        self.W_f = nn.Linear(xemb_size, h_size, bias=False)
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)
        self.b_f = nn.Parameter(torch.zeros(1, h_size))

    def message_func(self, edges):
        return {
            "h": edges.src["h"],
            "c": edges.src["c"],
            "type_n": edges.src["type_n"],
        }

    def reduce_func(self, nodes):

        c_child = nodes.mailbox["c"]  # (Nt, Nchn, H)
        h_child = nodes.mailbox["h"]  # (Nt, Nchn, H)
        childrn_num = c_child.size(1)
        hidden_size = c_child.size(2)

        # Step 1
        type_n = nodes.mailbox["type_n"]  # (Nt)
        type_n0_id = type_n == 0
        type_n1_id = type_n == 1

        # 1.b: creat mask matrix with the same size of h and c with zeros at
        # either type_0 node ids or type_1 node ids
        mask = torch.zeros_like(h_child)
        mask[type_n0_id] = 1  # mask one at type_0 nodes
        ht_0 = mask * h_child  # (Nt, Nchn, H)
        ht_0 = torch.sum(ht_0, dim=1)  # sum over child nodes => (Nt, H)

        mask = torch.zeros_like(h_child)  # do the same for type_1
        mask[type_n1_id] = 1
        ht_1 = mask * h_child  # (Nt, Nchn, H)
        ht_1 = torch.sum(ht_1, dim=1)  # sum over child nodes => (Nt, H)

        # # Step 2
        h_iou = torch.cat((ht_0, ht_1), dim=1)  # (Nt, 2H)

        # Step 3
        # (Nt, 2H) => (Nt, 2, H)
        f = self.U_f(torch.cat((ht_0, ht_1), dim=1)).view(-1, 2, hidden_size)
        # 3.b select from f either f_0 or f_1 using type_n as index
        # generate array repeating elements of nodes_id by their number of
        # children. e.g. if we have 3 nodes that have 2 children.
        # select_id = [0, 0, 1, 1, 2, 2]
        select_id = np.repeat(range(c_child.size(0)), c_child.size(1))
        f_cell = f[select_id, type_n.view(-1), :].view(*c_child.size())

        # Steps 3.c,d
        X = self.W_f(nodes.data["emb"])  # (Nt, H)
        X = X.repeat(childrn_num, 1).view(*c_child.size())  # (Nt, Nchn, H)
        f_tk = torch.sigmoid(X + f_cell + self.b_f)  # (Nt, Nchn, H)
        c_cell = torch.sum(f_tk * c_child, dim=1)  # (Nt, H)

        return {"h": h_iou, "c": c_cell}

    def apply_node_func(self, nodes):
        # The leaf nodes have no child the h_child is initialized.
        h_cell = nodes.data["h"]
        c_cell = nodes.data["c"]

        # Initialization for leaf nodes
        if nodes._graph.srcnodes().nelement() == 0:  # leaf nodes
            # initialize h states, for node type-0 and node type-1
            # NOTE: initialization for node type-0 == node type-1
            h_cell = torch.cat((h_cell, h_cell), dim=1)  # (Nt, Nchn*H)

        iou = self.W_iou(nodes.data["emb"]) + self.U_iou(h_cell) + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)  # (Nt x H) for each of i,o,u
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = i * u + c_cell
        h = o * torch.tanh(c)

        return {"h": h, "c": c}


class TypeTreeLSTM(nn.Module):

    """
    A. Reference:
    --------------

        This code is based on DGL's tree-LSTM implementation found in the paper [3]
        DGL
        Implementation can be found at
        https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py

    B. Papers list:
    ---------------

        1. Neural End-to-End Learning for Computational Argumentation Mining.
        https://arxiv.org/abs/1704.06104v2

        2. End-to-End Relation Extraction using LSTMs on Sequences and Tree
        Structures
        https://arxiv.org/abs/1601.00770

        3. Improved Semantic Representations From Tree-Structured Long Short-Term
        Memory Networks.
        https://arxiv.org/abs/1503.00075

        4. A Shortest Path Dependency Kernel for Relation Extraction
        https://dl.acm.org/doi/10.3115/1220575.1220666

    C. General Description:
    -----------------------

        This code implements the LSTM-ER model in [1], based on [2], to classify
        relations between argument components in a document. The LSTM-ER is derived
        from the N-ary treeLSTM architecture found in [3].  The relation extraction
        (RE) module in [2] utilizes the treeLSTM to process a sentence over its
        dependency tree. The dependency tree in nature has a varying number of
        children. However, N-ary design needs a fixed number of child nodes. For
        example, in [3], the N-ary tree is used with constituency binary-tree where
        each node has a left and a right child node.

        In [2], the dependency tree nodes are categorized into two classes: the
        shortest-path nodes are one class, and the other nodes are the second class.
        Nodes that belong to the same class share the same weights. As indicated in
        [4], the shortest path between two entities in the same sentence contains
        all the  information required to identify those relationships. However, as
        stated in [1], 92% of argument components' relationships are across
        different sentences. We cannot identify the path that encodes all the
        required information to identify the relationships between two arguments
        components across different sentences.

    D. Abbreviation:
    ---------------------
        B      : Batch size
        H      : LSTM's Hidden size
        E      : Embedding size, feature size
        SEQ    : Max Sequence length in the batch
        SEQ_S  : Max Sequence length per sentence
        SNT_N  : Max Number of sentences in the batch
        NE-OUT : Output dimension for NER module
        Nt     : total number of nodes (Nt) in the batch
        Nchn   : Number of children nodes in the batch
        DEP    : Dependency embedding size

    E. TreeLSTMCell Impelementation:
    --------------------------------
        For each parent node, `message_func` sends the children's information to
        `reduce_func` function . The following info is sent:
        h:      child nodes' hiddens state      Size: (Nt, Nchn, H)
        c:      child nodes' cell state         Size: (Nt, Nchn, H)
        type_n: child nodes' type               Size: (Nt, Nchn)

        The data is retained in `nodes.mailbox`. The return of `reduce_func`
        function is then sent to the next function, `apply_node_func`.

        We receive h and c in a tensor of size (Nt, Nchn, H). Because the
        number of children in the batch may vary, the `reduce_function`
        collects/groups the information according to the `Nchn` dim. It
        calls itself iteratively to process each group separately.  The
        function then stacks the results vetically and sends them. Thus,
        the dimensions other than Dimension(0) (i.e Nt) must be equal to each
        other. Also, the final number of rows, Dimension(0), after stacking must be
        equal to the number of nodes (batch size); i.e. Nt = number of parent nodes.

        For the leaf nodes, where there is no childen, the code starts at
        `apply_node_func`, The hidden state is initialized, then the the gates
        values are calculated

        E1. The forget gate eqn:
        -----------------------
            Assuming the following:
            1. For nodes in a graph [Ng], the number of nodes = n
            2. For node-t ∈ Ng & 1<=t<=N:
                a. Child nodes of node-t is [Nct]

                b. number of children of node-t: Nchn(t) = ℓ,
                For an arbitry node (node-r), r ≠ t and r ∈ [Ng]: Nchn(r) may not
                be equal to ℓ

                c. the hidden states for the child nodes htl = [hi] where
                1 <= i <= ℓ.
                Each child node is either of type_n0 or type_n1;
                the hidden state for typn_0 is h_t0 and for type_n1 is
                h_t1, where
                h_t0 = Sum([h0])= Sum( [hi | 1 <= j <= ℓ & m(j)=type_n0] )
                h_t1 = Sum([h1])= Sum( [hi | 1 <= j <= ℓ & m(j)=type_n1] )

                e. Node-t have ℓ forget gates; a gate for each child node

            In [1] eqn 4, the second part of the forget gate (Sum(U*h)) could
            be written as follows:
                - For each node-k in the child nodes: The forget gate
                (ftk, 1 <= k <= ℓ) is
                either a type_0 (f0) or (f1).  where:
                f0 = U00 h_t0 + U01 h_t1,  eq(a)
                f1 = U10 h_t0 + U11 h_t1   eq(b)

        E2. i,o,u eqn:
        --------------
            For node_t:
            i_t = U_i0 . h_t0 + U_i1 . h_t1   eq(c)
            o_t = U_o0 . h_t0 + U_o1 . h_t1   eq(d)
            u_t = U_u0 . h_t0 + U_u1 . h_t1   eq(e)

        E3. Example:
        -------------
            - Assuming a node-t = node-1 in a graph:
            - node-1 have 4 child nodes: Nct=[n1, n2, n3, n4].
            - The types of child nodes are as follows [0, 1, 1, 0]
            - Ignoring the fixed parts in the forget gates' equation: Wx & b:
                * the forget gate for each child node will be as follows:
                    For node-k that is child of node-t:
                    ftk = Um(tk)m(tl) * htl,
                    where: tl ∈ Nct, 1 <= tl < 4 & m(lt)=is either 0 or 1
            - For each child node, the equations are:
                child-node-1: f11 = U00 h11 + U01 h12 + U01 h13 + U00 h14
                child-node-2: f12 = U10 h11 + U11 h12 + U11 h13 + U10 h14
                child-node-3: f13 = U10 h11 + U11 h12 + U11 h13 + U10 h14
                child-node-4: f14 = U00 h11 + U01 h12 + U01 h13 + U00 h14
                child-node-5: f15 = U10 h11 + U11 h12 + U11 h13 + U10 h14

            - The equation of child-node 1,4 (type_n0) are equal to each
                other, the same are for child nodes 2,3, (type_n1).

            - Further reduction can be done as follows:
                forget type_0: f0 = U00 (h11 + h14) + U01 (h12 + h13)
                forget type_1: f1 = U10 (h11 + h14) + U11 (h12 + h13)
                h_t0 = (h11 + h14)
                h_t1 = (h12 + h13), see section E1.c above.

                f0 = U00 h_t0 + U01 h_t1
                f1 = U10 h_t0 + U11 h_t1
                where ht_0 is hidden states for type_n0 child nodes and ht_1 is
                hidden states for type_n1 child nodes.

        E4. Impelemntation:
        --------------------
            Step:1 Get ht_0 anf ht_1:
            *************************
                1. Get hidden states for each node type: ht_0, ht_1
                    a. Get nodes that are belong to each node type
                        (type: 0 & 1)
                    b. Get h and c for each node type "ht_0, ht_1"
                    c. If there is no specific node type,
                        the respective ht_0 or ht_1 is zeros

            Step:2 i,o,t gates: based on eqs(c,d,e) Under section D:
            **************************************************
                a. [ht_0, ht_1] [   Uiot   ] = [i, o, t]
                    (Nt , 2H)   (2H , 3H)   = (Nt , 3H)

                b. `reduce_func` return [i, o, t]

            Step:3 Forget gate: based on eqs(a,b) Under section C:
            ************************************************
                a. [ht_0, ht_1] [    Uf    ] =  [f0, f1]
                    (Nt , 2H)     (2H , 2H)  =  (Nt , 2H)

                b. Then, construct a tensor f_cell (Nt, Nchn, H) ,
                    where each tensor at (Nt, Nchn) is either
                    f_0 or f_1 according to the type of the respective
                    child node. for the example in section C the matrix
                    f_cell (1, 4, H) = [f0; f1; f1; f0]

                c. f_tk = sigma( W X_emb + f_cell + b)
                    The size of f_tk, [W X_emb] and f_cell = (Nt, Nchn, H)
                    The size of b is (1, H)

                d. c_cell = SUM(mailbox(c) . f_tk) over Dimension(Nchn)
                    The size of c mailbox(c) = size of f_tk
                    c_cell size = (Nt, H)

                e. return c_cell

    """

    def __init__(
        self,
        embedding_dim,
        h_size,
        dropout=0,
        bidirectional=True,
    ):

        super(TypeTreeLSTM, self).__init__()

        self.bidirectional = bidirectional
        self.TeeLSTM_cell = TreeLSTMCell(embedding_dim, h_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, g: DGLGraph, h0: Tensor, c0: Tensor):
        """A modified N-ary tree-lstm (LSTM-ER) network
        ----------
        g:      dgl.DGLGraph
                Batch of Trees for computation
        h0:     Tensor
                Initial hidden state.
        c0:     Tensor
                Initial cell state.
        Returns
        -------
        logits: Tensor

        """

        # Tree-LSTM (LSTM-ER) according to arXiv:1601.00770v3 sections 3.3 & 3.4
        g.ndata["h"] = h0
        g.ndata["c"] = c0

        # copy graph
        if self.bidirectional:
            g_copy = g.clone()

        # propagate bottom top direction
        dgl.prop_nodes_topo(
            g,
            message_func=self.TeeLSTM_cell.message_func,
            reduce_func=self.TeeLSTM_cell.reduce_func,
            apply_node_func=self.TeeLSTM_cell.apply_node_func,
        )
        logits = g.ndata.pop("h")

        if self.bidirectional:
            # propagate top bottom direction
            dgl.prop_nodes_topo(
                g_copy,
                message_func=self.TeeLSTM_cell.message_func,
                reduce_func=self.TeeLSTM_cell.reduce_func,
                apply_node_func=self.TeeLSTM_cell.apply_node_func,
                reverse=True,
            )
            logits_tb = g_copy.ndata.pop("h")

            # concatenate both tree directions
            logits = torch.cat((logits, logits_tb), dim=-1)

        return logits


class DepGraph:

    def __init__(self,
                 token_embs: Tensor,
                 deplinks: Tensor,
                 roots: Tensor,
                 token_mask: Tensor,
                 starts: Tensor,
                 ends: Tensor,
                 lengths: Tensor,
                 mode: str,
                 device=None,
                 assertion: bool = False
                 ) -> List[DGLGraph]:

        assert mode in set([
            "shortest_path"
        ]), f"{mode} is not a supported mode for DepPairingLayer"
        self.device = device
        batch_size = deplinks.size(0)

        # # creat sample graphs G(U, V) tensor on CPU
        # U, V, M = self.get_sample_graph(
        #                                 deplinks=deplinks,
        #                                 roots=roots,
        #                                 token_mask=token_mask,
        #                                 lengths = lengths,
        #                                 assertion=assertion
        #                                 )

        # creat sub_graph for each pair
        dep_graphs = []
        nodes_emb = []
        
        starts = torch.split(starts, lengths, dim=0)
        ends  = torch.split(ends, lengths, dim=0)

        #token_mask = token_mask.type(torch.bool)

        for b_i in range(batch_size):

            if lengths[b_i] == 0:  # no candidate pair
                continue
            
            #create v and u from deplinks and nr of tokens in sample
            v = deplinks[b_i][token_mask[b_i]]
            u = torch.arange(len(v), device=device)


            ## filter out self loops at root noodes, THIS BECAUSE?
            self_loop = u == v
            u = u[~self_loop]
            v = v[~self_loop]

            # creat sample DGLGraph, convert it to unidirection, separate the
            # list of tuples candidate pairs into two lists: (start and end
            # tokens), then create subgraph
            graph = dgl.graph((u,v))

    
            graph_unidir = graph.to_networkx().to_undirected()

            start = utils.ensure_numpy(starts[b_i])
            end = utils.ensure_numpy(ends[b_i])

            subgraph_func = functools.partial(
                                            self.get_subgraph,
                                            g=graph,
                                            g_nx=graph_unidir,
                                            sub_graph_type=mode,
                                            assertion=assertion
                                            )

            dep_graphs.append(dgl.batch(list(map(subgraph_func, start, end))))

            # get nodes' token embedding
            nodes_emb.append(token_embs[b_i, dep_graphs[-1].ndata["_ID"]])


        # batch graphs, move to model device, update nodes data by tokens
        # embedding
        nodes_emb = torch.cat(nodes_emb, dim=0)
        self.graphs = dgl.batch(dep_graphs).to(self.device)
        self.graphs.ndata["emb"] = nodes_emb

    # def assert_graph(self, u: Tensor, v: Tensor, roots: Tensor,
    #                  self_loop: Tensor) -> None:

    #     # check that self loops do not exist in places other than roots
    #     self_loop_check = u == roots[:, None]
    #     if not torch.all(self_loop == self_loop_check):
    #         # we have a problem. Get samples ids where we have more than one
    #         # self loop
    #         problem_id = torch.where(torch.sum(self_loop, 1) > 1)[0]
    #         self_loop_id = v[problem_id, :][self_loop[problem_id, :]].tolist()
    #         theroot = roots[problem_id].tolist()
    #         error_msg_1 = f"Self loop found in sample(s): {problem_id}, "
    #         error_msg_2 = f"Node(s) {self_loop_id}. Root(s): {theroot}."
    #         raise Exception(error_msg_1 + error_msg_2)


    # def get_sample_graph(self, 
    #                     deplinks: Tensor, 
    #                     roots: Tensor,
    #                     token_mask: Tensor,
    #                     lengths: Tensor,
    #                     assertion: bool
    #                     ) -> List[Tensor]:

    #     batch_size = deplinks.size(0)
    #     max_lenght = deplinks.size(1)
    #     device = torch.device("cpu")

    #     # G(U, V)
    #     U = torch.arange(max_lenght, device=self.device).repeat(batch_size, 1)
    #     V = deplinks.clone()
    #     M = token_mask.clone()

    #     # remove self loops at root nodes
    #     self_loop = U == V
    #     if assertion:
    #         self.assert_graph(U, V, roots, self_loop)

    #     U = torch.split(U[~self_loop].to(device), lengths)
    #     V = torch.split(V[~self_loop].to(device), lengths)
    #     M = torch.split(M[~self_loop].to(device), lengths)

    #     return U, V, M

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
                 input_size: int,
                 hidden_size: int,
                 bidir: bool,
                 mode : str = "shortest_path",
                 dropout: int = 0.0
                 ):
        super().__init__()

        self.mode = "shortest_path"

        self.hidden_size = hidden_size
        self.tree_lstm_bidir = bidir

        self.tree_lstm = TypeTreeLSTM(
                                        embedding_dim=input_size,
                                        h_size=hidden_size,
                                        dropout=dropout,
                                        bidirectional=True
                                      )
        
        self.output_size = hidden_size * 3


    def split_nested_list(self,
                          nested_list: List[list],
                          device=torch.device
                          ) -> Tuple[Tensor]:

        list_1, list_2 = list(zip(*chain.from_iterable(nested_list)))
        list_1 = torch.tensor(list_1, dtype=torch.long, device=device)
        list_2 = torch.tensor(list_2, dtype=torch.long, device=device)
        return list_1, list_2


    def forward(self,
                input: Union[Tensor, Sequence[Tensor]],
                roots: Tensor,
                token_mask: Tensor,
                deplinks: Tensor,
                starts: Tensor,
                ends: Tensor,
                lengths: Tensor,
                assertion: bool = False
                ):


        if not isinstance(input, Tensor):
            input = torch.cat(input, dim=-1)
        
        device = input.device

        # 1) Build graph from dependecy data
        G = DepGraph(
                    token_embs = input,
                    deplinks = deplinks,
                    roots = roots,
                    token_mask = token_mask,
                    starts = starts,
                    ends = ends,
                    lengths = lengths, #lengths in pairs per sample
                    mode = self.mode,
                    device = device,
                    assertion = assertion
                    )

        # 2) Pass graph to a TreeLSTM to create hidden representations
        h0 = torch.zeros(
                         G.graphs.num_nodes(),
                         self.hidden_size,
                         device = device
                         )

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

        tree_lstm_out = tree_lstm_out.view(-1, 2, self.hidden_size)
        tree_pair_embs = torch.cat(
                    (
                        tree_lstm_out[root_id, 0, :],  # ↑hpA
                        tree_lstm_out[start_id, 1, :],  # ↓hp1
                        tree_lstm_out[end_id, 1, :]  # ↓hp2
                    ),
                    dim=-1)

        return tree_pair_embs



