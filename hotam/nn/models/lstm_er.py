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

from collections import defaultdict
import itertools as it
from typing import List, Dict, Tuple

from math import exp
from random import random

import numpy as np
import torch as th
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
import dgl
from dgl import heterograph
from dgl.traversal import topological_nodes_generator as traverse_topo

from hotam.nn.layers.lstm import LSTM_LAYER


class Graph():
    def __init__(
        self,
        V: Tensor,
        lengthes: List[List],
        pad_mask: Tensor,
        roots_id: Tensor,
        graph_buid_type=0,
    ) -> None:
        """Construct a batch of graphs form token head tensor
        V:                  Tensor, (B, SNT_N, SEQ_S)
                            Destination nodes, depedency head

        lengthes:           List[List]
                            Lengthes of sentences

        pad_mask:           Tensor (B, SEQ)
                            Mask od pad tokens

        roots_id:           Tensor (B, SNT_N)
                            token_id of the root of each sentence

        graph_buid_type:    int
                            Type of the graphes to be build

        """
        self.roots_g = []
        self.G = []
        device = th.device("cpu")

        V = V.detach().cpu()
        pad_mask = pad_mask.detach().cpu()
        roots_id = roots_id.detach().cpu()
        for i, data in enumerate(zip(lengthes, V, roots_id, pad_mask)):
            lens_, v_i, root_id, p_mask = data
            lens = th.tensor(lens_ + [-1] * (v_i.size(0) - len(lens_)))
            mask = lens[:, None] > th.arange(v_i.size(1))
            # token_ids of the depdency heads are w.r.t. sentence. So, in v_i
            # token_id = 1, could be in sentence_1 or sentence_2. However, we
            # deal with ids w.r.t. sample, meaning that id sentence_0 has 5
            # tokens, then token_id=0 in sentence_1 will be 6
            # Calculate Acculmulative sum for length, to get absolute ids
            lens_accum = th.cumsum(th.tensor(lens), dim=0)
            delta2abs = lens_accum[:-1]  # (SNT_N - 1)
            v_i_abs = v_i + th.cat((th.tensor([0]), delta2abs))[:, None]

            v = th.masked_select(v_i_abs, mask=mask)  # select non pad token
            u = th.arange(th.squeeze(mask).sum(), dtype=th.long, device=device)

            # convert root_id to absolute ids w.r.t sample just as we did with
            # v_i
            roots_abs = root_id + th.cat((th.tensor([0]), delta2abs))
            roots_abs = roots_abs[:len(lens_)]  # select based on sent length
            # Connect sentences in each sample sample
            u_g, v_g, roots_g = self.connect_sents(u, v, roots_abs,
                                                   lens_accum[:len(lens_)],
                                                   graph_buid_type)
            # create graph and initialize nodes data
            g = dgl.graph((u_g, v_g), device=device)
            # get ids for non pad tokens
            root_num = len(list(traverse_topo(g))[-1])
            assert root_num == 1

            self.G.append(g)  # type: heterograph
            self.device = device

        # self.G = dgl.batch(all_graph)
        self.mask = pad_mask  # (B, n_tokens)
        # self.nonpad_idx = th.squeeze(th.nonzero(pad_mask.view(-1)))
        # self.batch_update_nodes(ndata=ndata)

    def connect_sents(self, u: Tensor, v: Tensor, roots: Tensor,
                      lengths_accum: Tensor,
                      graph_buid_type: int) -> Tuple[Tensor, Tensor, Tensor]:
        if graph_buid_type == 0:
            # connect the root of the current sent to the end of the prev sent
            v[roots[1:]] = lengths_accum[:-1] - 1
            # remove the self connection of the first root
            u = th.cat((u[:roots[0]], u[roots[0] + 1:]))
            v = th.cat((v[:roots[0]], v[roots[0] + 1:]))
            # self.token_rmvd.append(roots[0].item())
            roots = roots[1:]
        return u, v, roots

    def __extract_graph(self, g, g_nx, start_n: list, end_n: list,
                        sub_graph_type: int):
        if sub_graph_type == 0:
            thepath = nx.shortest_path(g_nx, source=start_n, target=end_n)
            sub_g = dgl.node_subgraph(g, thepath)
            root = list(traverse_topo(sub_g))[-1]
            assert len(root) == 1
            sub_g.ndata["type_n"] = th.zeros(sub_g.number_of_nodes(),
                                             device=self.device)
        elif sub_graph_type == 1:
            # get subtree
            pass
        return sub_g, root.item()

    def get_subgraphs(
        self,
        starts: list,
        ends: list,
        batch_num: int,
        sub_graph_type: int = 0,
    ):
        g = self.G[batch_num]
        g_nx = g.to_networkx().to_undirected()
        graphs_sub = []
        roots = []
        for src, dst in zip(starts.tolist(), ends.tolist()):
            g_sub, root = self.__extract_graph(g, g_nx, src, dst,
                                               sub_graph_type)
            g_sub_r, root_r = self.__extract_graph(g, g_nx, dst, src,
                                                   sub_graph_type)
            graphs_sub.extend([g_sub, g_sub_r])
            roots.extend([root, root_r])

        return dgl.batch(graphs_sub), roots

    def update_batch(self, ndata_dict: Dict[str, Tensor]) -> None:
        """update graph node data
        """
        G_batch = dgl.batch(self.G)
        for n_data_name in ndata_dict:
            # node data size is either (B, SEQ, embedding_size) or (B, SEQ)
            # Change dim to (B*SEQ, embedding_size) or (B*SEQ), then select non
            # pad tokens Batch graph, updat node data, unbatch again
            n_data = ndata_dict[n_data_name]
            emb_size = n_data.size(-1)
            dim_3d = bool(len(n_data.size()) == 3)
            n_data = n_data.view(-1, emb_size) if dim_3d else n_data.view(-1)
            n_data = n_data[self.mask.view(-1), :]
            G_batch.ndata[n_data_name] = n_data

        self.G = dgl.unbatch(G_batch)


class TreeLSTMCell(nn.Module):
    def __init__(self, xemb_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(xemb_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))

        self.W_f = nn.Linear(xemb_size, h_size, bias=False)
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)
        self.b_f = nn.Parameter(th.zeros(1, h_size))

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
        mask = th.zeros((*h_child.size()))
        mask[type_n0_id] = 1  # mask one at type_0 nodes
        ht_0 = mask * h_child  # (Nt, Nchn, H)
        ht_0 = th.sum(ht_0, dim=1)  # sum over child nodes => (Nt, H)

        mask = th.zeros((*h_child.size()))  # do the same for type_1
        mask[type_n1_id] = 1
        ht_1 = mask * h_child  # (Nt, Nchn, H)
        ht_1 = th.sum(ht_1, dim=1)  # sum over child nodes => (Nt, H)

        # # Step 2
        h_iou = th.cat((ht_0, ht_1), dim=1)  # (Nt, 2H)

        # Step 3
        # (Nt, 2H) => (Nt, 2, H)
        f = self.U_f(th.cat((ht_0, ht_1), dim=1)).view(-1, 2, hidden_size)
        # 3.b select from f either f_0 or f_1 using type_n as index
        # generate array repeating elements of nodes_id by their number of
        # children. e.g. if we have 3 nodes that have 2 children.
        # select_id = [0, 0, 1, 1, 2, 2]
        select_id = np.repeat(range(c_child.size(0)), c_child.size(1))
        f_cell = f[select_id, type_n.view(-1), :].view(*c_child.size())

        # Steps 3.c,d
        X = self.W_f(nodes.data["emb"])  # (Nt, H)
        X = X.repeat(childrn_num, 1).view(*c_child.size())  # (Nt, Nchn, H)
        f_tk = th.sigmoid(X + f_cell + self.b_f)  # (Nt, Nchn, H)
        c_cell = th.sum(f_tk * c_child, dim=1)  # (Nt, H)

        return {"h": h_iou, "c": c_cell}

    def apply_node_func(self, nodes):
        # The leaf nodes have no child the h_child is initialized.
        h_cell = nodes.data["h"]
        c_cell = nodes.data["c"]

        # Initialization for leaf nodes
        if nodes._graph.srcnodes().nelement() == 0:  # leaf nodes
            # initialize h states, for node type-0 and node type-1
            # NOTE: initialization for node type-0 == node type-1
            h_cell = th.cat((h_cell, h_cell), dim=1)  # (Nt, Nchn*H)

        iou = self.W_iou(nodes.data["emb"]) + self.U_iou(h_cell) + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)  # (Nt x H) for each of i,o,u
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)

        c = i * u + c_cell
        h = o * th.tanh(c)

        return {"h": h, "c": c}


class TreeLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim,
        h_size,
        dropout=0,
        bidirectional=True,
    ):

        super(TreeLSTM, self).__init__()

        self.bidirectional = bidirectional
        self.TeeLSTM_cell = TreeLSTMCell(embedding_dim, h_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, g, roots, leaves, h, c):
        """A modified N-ary tree-lstm (LSTM-ER) network
        ----------
        g :     dgl.DGLGraph
                Batch of Trees for computation
        roots:  list
                tree root id for each tree in the graph batch
        leaves: list
                leaves id for each tree in the graph batch
        h :     Tensor
                Initial hidden state.
        c :     Tensor
                Initial cell state.
        Returns
        -------
        logits_bt : Tensor
                    The hidden state of bottom-up direction =
                    trees roots' hidden state.
        logits_tb : Tensor
                    The hidden state of of up-bottom direction =
                    leaves nodes' hidden state
        """

        # Tree-LSTM (LSTM-ER) according to arXiv:1601.00770v3 sections 3.3 & 3.4
        g.ndata["h"] = h
        g.ndata["c"] = c

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
        logits_bt = g.ndata.pop("h")[roots, :]

        if self.bidirectional:
            # propagate top bottom direction
            dgl.prop_nodes_topo(
                g_copy,
                message_func=self.TeeLSTM_cell.message_func,
                reduce_func=self.TeeLSTM_cell.reduce_func,
                apply_node_func=self.TeeLSTM_cell.apply_node_func,
                reverse=True,
            )
            logits_tb = g_copy.ndata.pop("h")[leaves, :]
            # concatenate both tree directions

        return logits_bt, logits_tb


class NELabelEmbedding(nn.Module):
    def __init__(self, encode_size):

        super(NELabelEmbedding, self).__init__()
        self.encode_size = encode_size

    def forward(self, prediction_id, device):
        # label prediction, one-hot encoding for label embedding
        batch_size = prediction_id.size(0)
        label_one_hot = th.zeros(batch_size, self.encode_size, device=device)
        label_one_hot[th.arange(batch_size), prediction_id] = 1

        return label_one_hot


class AC_Seg_Module(nn.Module):
    def __init__(
        self,
        token_embedding_size,
        label_embedding_size,
        h_size,
        ner_hidden_size,
        ner_output_size,
        bidirectional=True,
        num_layers=1,
        dropout=0,
    ):
        super(AC_Seg_Module, self).__init__()

        self.model_param = nn.Parameter(th.empty(0))
        self.label_embedding_size = label_embedding_size

        # Entity label prediction according to arXiv:1601.00770v3, section 3.4
        # Sequential LSTM layer
        self.seqLSTM = LSTM_LAYER(
            input_size=token_embedding_size,
            hidden_size=h_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)

        # Entity prediction layer: two layers feedforward network
        # The input to this layer is the the predicted label of the previous
        # word and the current hidden state.
        num_dirs = 2 if bidirectional else 1
        ner_input_size = label_embedding_size + (h_size * num_dirs)
        self.ner_decoder = nn.Sequential(
            nn.Linear(ner_input_size, ner_hidden_size),
            nn.Tanh(),
            self.dropout,
            nn.Linear(ner_hidden_size, ner_output_size),
        )

        # self.label_embs_size = label_embedding_size
        self.entity_embedding = NELabelEmbedding(
            encode_size=label_embedding_size)

    def forward(self,
                batch_embedded,
                lengths,
                mask,
                pad_id=0.0,
                return_type=0,
                h=None,
                c=None):
        """
        ----------


        Returns
        -------

        """
        device = self.model_param.device

        # LSTM layer
        # ============
        # sort batch according to the sample length
        batch_embedded = self.dropout(batch_embedded)
        batch_embedded = batch_embedded.rename(None)
        lengths_sorted, ids_sorted = th.sort(lengths, descending=True)
        _, ids_original = th.sort(ids_sorted, descending=False)

        if h is not None and c is not None:
            lstm_out, _ = self.seqLSTM(
                (batch_embedded[ids_sorted], h.detach(), c.detach()),
                lengths_sorted)
        else:
            lstm_out, _ = self.seqLSTM(batch_embedded[ids_sorted],
                                       lengths_sorted)
        lstm_out = lstm_out[ids_original]

        # Entity prediction layer
        logits = []  # out_list_len=SEQ, inner_list_of_list_len=(B, NE-OUT)
        prob_dis = []
        label_id_predicted = []  # list_of_list_len=(SEQ, 1)
        label_predicted_embs = []

        # construct initial previous predicted label, v_{t-1}:v_0
        batch_size = lstm_out.size(0)
        seq_length = lstm_out.size(1)
        v_t_old = th.zeros(batch_size,
                           self.label_embedding_size,
                           device=device)
        for i in range(seq_length):  # loop over words
            # construct input, get logits for word_i, softmax, entity prediction
            ner_input = th.cat(
                (lstm_out[:, i, :].view(batch_size, -1), v_t_old), dim=1)
            # QSTN is this order is correct? Is the dim is correct? what is
            # dim=-1?
            logits_i = self.ner_decoder(ner_input)  # (B, NE-OUT)
            prob_dist = F.softmax(logits_i, dim=1)  # (B, NE-OUT)
            prediction_id = th.max(prob_dist, dim=1)[1]

            # entity label embedding
            label_one_hot = self.entity_embedding(prediction_id, device)
            label_one_hot = self.dropout(label_one_hot)
            label_predicted_embs.append(label_one_hot.detach().tolist())
            # save data
            label_id_predicted.append(prediction_id.detach().tolist())
            logits.append(logits_i.detach().tolist())
            prob_dis.append(prob_dist.detach().tolist())
            v_t_old = label_one_hot  # v_{t-1} <- v_t

        # Reshape logits dimension from (SEQ, B, NE-OUT) to (B, SEQ, NE-OUT)
        # Reshape label_id_predicted from (SEQ, B) to (B, SEQ)
        logits = th.tensor(logits,
                           device=device,
                           dtype=th.float,
                           requires_grad=True).view(batch_size, seq_length, -1)
        prob_dis = th.tensor(prob_dis,
                             device=device,
                             dtype=th.float,
                             requires_grad=True).view(batch_size, seq_length,
                                                      -1)
        label_id_pred = th.tensor(label_id_predicted,
                                  device=device,
                                  dtype=th.float).view(batch_size, -1)
        label_id_pred_embs = th.tensor(label_predicted_embs,
                                       device=device,
                                       dtype=th.float).view(
                                           batch_size, seq_length, -1)
        # TODO return type: what to return according to the input return_type
        return logits, prob_dis, label_id_pred, label_id_pred_embs, lstm_out


class LSTM_RE(nn.Module):
    def __init__(self, hyperparamaters: dict, task2labels: dict,
                 feature2dim: dict):

        super(LSTM_RE, self).__init__()

        num_ne = len(task2labels["seg_ac"])  # number of named entities
        num_relations = len(task2labels["stance"])  # number of relations
        self.last_tkn_data = last_word_pattern(task2labels["seg_ac"])

        self.model_param = nn.Parameter(th.empty(0))

        self.graph_buid_type = hyperparamaters["graph_buid_type"]
        self.sub_graph_type = hyperparamaters["sub_graph_type"]

        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.BATCH_SIZE = hyperparamaters["batch_size"]

        self.k = hyperparamaters["k"]

        # Embed dimension for tokens
        token_embs_size = feature2dim["word_embs"] + feature2dim["pos_embs"]
        # Embed dimension for entity labels
        label_embs_size = num_ne + 1
        # Embed dimension for dependency labels
        dep_embs_size = feature2dim["deprel_embs"]
        # Sequential LSTM hidden size
        seq_lstm_h_size = hyperparamaters["seq_lstm_h_size"]
        # Tree LSTM hidden size
        tree_lstm_h_size = hyperparamaters["tree_lstm_h_size"]
        # Entity recognition layer hidden size
        ner_hidden_size = hyperparamaters["ner_hidden_size"]
        # Entity recognition layer output size
        ner_output_size = num_ne
        # Relation extraction layer hidden size
        re_hidden_size = hyperparamaters["re_hidden_size"]
        # Relation extraction layer output size
        re_output_size = num_relations
        # Sequential LSTM number of layer
        seq_lstm_num_layers = hyperparamaters["seq_lstm_num_layers"]
        # Sequential LSTM bidirection
        lstm_bidirectional = hyperparamaters["lstm_bidirectional"]
        # Tree LSTM bidirection
        tree_bidirectional = hyperparamaters["tree_bidirectional"]
        dropout = hyperparamaters["dropout"]

        # Argument module
        self.module_ac_seg = AC_Seg_Module(
            token_embedding_size=token_embs_size,
            label_embedding_size=label_embs_size,
            h_size=seq_lstm_h_size,
            ner_hidden_size=ner_hidden_size,
            ner_output_size=ner_output_size,
            bidirectional=lstm_bidirectional,
            num_layers=seq_lstm_num_layers,
            dropout=dropout)

        # Relation extraction module
        nt = 3 if tree_bidirectional else 1
        ns = 2 if lstm_bidirectional else 1
        re_input_size = tree_lstm_h_size * nt
        tree_input_size = seq_lstm_h_size * ns + dep_embs_size + label_embs_size
        self.module_re = nn.Sequential(
            TreeLSTM(embedding_dim=tree_input_size,
                     h_size=tree_lstm_h_size,
                     dropout=dropout,
                     bidirectional=tree_bidirectional),
            nn.Linear(re_input_size, re_hidden_size), nn.Tanh(),
            nn.Linear(re_hidden_size, re_output_size))

        self.label_one_hot = NELabelEmbedding(label_embs_size)

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

    @classmethod
    def name(self):
        return "LSTM_ER"

    def forward(self, batch):
        """Compute logits of and predict argument components label.
        ----------


        Returns
        -------

        """
        device = self.model_param.device
        d_cpu = th.device('cpu')

        # Batch data:
        # ================
        # Embeddings:
        # ----------
        token_embs = batch["word_embs"]  # Tensor["B", "SEQ", "E"]
        pos_embs = batch["pos_embs"]  # Tensor["B", "SEQ", "E"]
        dep_embs = batch["deprel_embs"]  # Tensor["B", "SEQ", "E"]

        # Token data:
        # ----------
        token_head = batch["dephead"]  # Tensor["B", "SENT_NUM", "SEQ"]
        pad_mask = batch["token_mask"]  # pad token mask. Tensor["B", "SEQ"]

        # Sample information:
        # -------------------
        lengths_sent_tok = batch["lengths_sent_tok"]  # type: list[list]
        lengths_per_sample = batch["lengths_tok"]  # type: list

        # NOTE sents_root dim1 is not correct, have to fixe it here
        sents_root = batch["sent2root"]  # Tensor[B, SENT_NUM]
        if (sents_root.size(1) >= token_head.size(1)):
            _SENT_NUM_ = token_head.size(1)
            sents_root = sents_root[:, :_SENT_NUM_]

        batch_size = token_embs.size(0)

        # AC Segmentation Module:
        # =======================
        # NOTE Is it important to initialize h, c?
        input_ac_seg = th.cat((token_embs, pos_embs), dim=2)
        ac_seg_pred = self.module_ac_seg(input_ac_seg, lengths_per_sample,
                                         pad_mask)
        logitss_seg, prob_seg, pred_seg, pred_embs_seg, h_seg = ac_seg_pred

        # Get relations
        # =============
        # all possible  relations in both directions between the last tokens
        # of the detected entities.
        schdule_sampling = self.k / (self.k +
                                     exp(batch.current_epoch / self.k))
        # schdule sampling
        coin_flip = random()
        if schdule_sampling > coin_flip:
            # use golden standard
            seg_ac_used = batch['seg_ac']
            # one hot encoding of golden standard
            seg_ac_used_embs = self.label_one_hot(seg_ac_used.view(-1), device)
            enc_size = seg_ac_used_embs.size(-1)
            seg_ac_used_embs = seg_ac_used_embs.view(batch_size, -1, enc_size)
            # assert th.equal(th.argmax(seg_ac_used_embs, dim=-1), seg_ac_used)

        else:
            seg_ac_used = pred_seg
            seg_ac_used_embs = pred_embs_seg

        # Build Graph for the batch:
        # ==========================
        graphs = Graph(V=token_head,
                       lengthes=lengths_sent_tok,
                       pad_mask=pad_mask,
                       roots_id=sents_root,
                       graph_buid_type=self.graph_buid_type)
        nodes_input = th.cat((h_seg, dep_embs, pred_embs_seg), dim=-1)
        # update graph data
        graphs.update_batch(ndata_dict={"emb": nodes_input})
        relations_data = sub_batch(seg_ac_used, pad_mask, self.last_tkn_data)
        # NOTE loop is needed because the sentence lengthes differ across
        rel_graphs = []
        rel_roots = []
        for r1, r2, sample_id in relations_data:
            # no relations, build empty graphs
            if r1.nelement() == 0:
                continue

            sub_graphs, roots = graphs.get_subgraphs(r1, r2, sample_id)
            rel_graphs.append(sub_graphs)
            rel_roots.append(roots)

        # Calculation of losses:
        # =======================
        # Entity Recognition Module.
        # (B, SEQ, NE-OUT) --> (B*SEQ, NE-OUT)
        logitss_seg = logitss_seg.view(-1, logitss_seg.size(-1))
        batch.change_pad_value(-1)  # ignore -1 in the loss function
        ground_truth_ner = batch["seg_ac"].view(-1)
        loss_seg = self.loss(logitss_seg, ground_truth_ner)

        # Relation Extraction Module.
        loss_total = loss_seg
        return {
            "loss": {
                "total": loss_total,
            },
            "preds": {
                "seg_ac": pred_seg,
                # "relation": stance_preds,
                # "stance": stance_preds
            },
            "probs": {
                "seg_ac": prob_seg,
                # "relation": relation_probs,
                # "stance": stance_probs
            },
        }


def last_word_pattern(ne_labels):
    """
    """
    # Get patterns of pair of labels that determine the end of a named entity.
    ne_label_data = defaultdict(list)
    for i, label in enumerate(ne_labels):
        ne_label_data[label[0]].append(i)

    B_B = list(map(list, it.product(ne_label_data["B"], repeat=2)))
    B_O = [[i, j] for i in ne_label_data["B"] for j in ne_label_data["O"]]
    IB_IO = [[i, j] for i in ne_label_data["I"]
             for j in ne_label_data["O"] + ne_label_data["B"]]

    B_Idash_I_Idash = []
    for i, labeli in enumerate(ne_labels):
        for j, labelj in enumerate(ne_labels):
            if labeli[0] == "B" and labelj[0] == "I" or \
               labeli[0] == "I" and labelj[0] == "I":
                if labeli[1:] != labelj[1:]:
                    B_Idash_I_Idash.append([i, j])

    return B_B + B_O + IB_IO + B_Idash_I_Idash


def sub_batch(predictions, token_mask, last_token_pattern: list):
    """
    """
    # 1. Get predicted entities' last token
    #    last_token_pattern contains a list of two consecutive labels that
    #    idintfy the last token. These pairs are as follows:
    #    [B, B], [B, O], [I, B], [I, O],
    #    [B-Claim, I-not-Claim]: any combination of B-I that are not belong to
    #    the same entity class
    #    [I-Claim, I-not-Claim]: ny combination of I-I that are not belong to
    #    the same entity class
    #    Each time one of these pattern appears in the label sequence, the first
    #    label of the pair is the last token.
    #
    #    For example, suppose that we have the following sequence of tags; the
    #    last tokens will be:
    #          ↓  ↓     ↓     ↓        ↓     ↓
    #    B  I  I  B  O  B  B  I  O  I  I  O  B  O  [B] <--- dummy label to pair
    #    ++++  ----  ++++  ++++  ++++  ----  ----           the last label
    #       ++++  ----  ----  ----  ++++  ++++  +++++
    #
    #    Last tokens are marked with ↓, last token pattern are marked with
    #    (---) other pairs are marked with (+++).
    #
    #    Implementation Logic:
    #    --------------------
    #    a. Shift right the predicted label, append [B] at the end
    #    b. Form a list of zip like pairs: c = torch.stack((a,b), dim=2)
    #       c has a shape of (B, SEQ, 2)
    #    c. Search for patterns; (c[:, :, None] == cmp).all(-1).any(-1)

    # 2. for each sample, get all posiible relation in both directions between
    #    last token

    # step: 1, mask last word by true
    d = th.device('cpu')
    last_token_pattern = th.tensor(last_token_pattern, dtype=th.long,
                                   device=d).view(-1, 2)
    B_lbl = th.tensor([last_token_pattern[0][0]] * predictions.size(0),
                      dtype=th.long,
                      device=d).view(-1, 1)
    p = predictions.detach().clone().cpu()
    p_rshifted = th.cat((p[:, 1:], B_lbl), dim=1)
    p_zipped = th.stack((p, p_rshifted), dim=2)
    last_word_mask = (p_zipped[:, :,
                               None] == last_token_pattern).all(-1).any(-1)

    # step: 2
    # NOTE For loop is needed is the number of items are different
    for i in range(last_word_mask.size(0)):
        # get the last token id in each sample
        # select rows in last_word_mask using idx
        # get the lat word id
        # idx_tensor = th.tensor(idx, dtype=th.long)  # sample id to tensor
        sample_mask = last_word_mask[i, :]
        sample_mask = sample_mask[token_mask[i, :]]
        last_word_id = th.nonzero(sample_mask).flatten()
        # get relations per sample
        # list of list of list. The relation in each sample is in list of
        # list.
        # For example, assume in sample-i there are relations betwenn
        # tokens-(1,4) and tokens-(5,9). Then the relation will be in form
        # of [[1, 5], [4, 9]]
        r = list(map(list, zip(*it.combinations(last_word_id.tolist(), 2))))
        r1 = th.tensor(r[0], device=d)
        r2 = th.tensor(r[1], device=d)
        yield r1, r2, i
