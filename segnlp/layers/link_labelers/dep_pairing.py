
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


class DepPairingLayer(nn.Module):
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
        super(DepPairingLayer, self).__init__()

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


    def swap_elements(self, vector_1: Tensor, vector_2: Tensor,
                      idx: Tensor) -> Tuple[Tensor]:
        vector_1_copy = vector_1.clone()
        vector_2_copy = vector_2.clone()  # avoid changing original tensor

        vector_1_copy[idx] = vector_2[idx]
        vector_2_copy[idx] = vector_1[idx]

        return vector_1_copy, vector_2_copy


    def split_2dtensor_start_end(self, matrix: Tensor, start_idx: np.array,
                                 end_idx: np.array, idx0: np.array):
        idx_0 = idx0.clone()
        span_length = end_idx - start_idx
        idx_0 = torch.repeat_interleave(idx_0, span_length)
        idx_1 = torch.cat(
            list(
                map(lambda x, y, d: torch.arange(x, y, device=d), start_idx,
                    end_idx, repeat(idx_0.device))))

        matrix_split = torch.split(matrix[idx_0, idx_1],
                                   split_size_or_sections=span_length.tolist())

        return matrix_split


    def build_pair_embs(self,
                        token_embs: Tensor,
                        dep_embs: Tensor,
                        one_hot_embs: Tensor,
                        roots: Tensor,
                        token_mask: Tensor,
                        deplinks: Tensor,
                        seg_data: dict,
                        assertion: bool = False):

        # get all possible pairs
        pair_data = get_all_possible_pairs(
                                        start=seg_data["unit"]["start"],
                                        end=seg_data["unit"]["end"],
                                        device=self.device,
                                        bidir=False
                                        )

        # 1) Build graph from dependecy data
        node_embs = torch.cat((token_embs, one_hot_embs, dep_embs), dim=-1)
        # start_graph_time = time.time()
        G = DepGraph(token_embs=node_embs,
                     deplinks=deplinks,
                     roots=roots,
                     token_mask=token_mask,
                     subgraphs=pair_data["end"],
                     mode=self.mode,
                     device=self.device,
                     assertion=assertion)
        # end_graph_time = time.time()

        # 2) Pass graph to a TreeLSTM to create hidden representations
        h0 = torch.zeros(G.graphs.num_nodes(),
                         self.tree_lstm_h_size,
                         device=self.device)
        c0 = torch.zeros_like(h0)
        # start_lstm_time = time.time()
        tree_lstm_out = self.tree_lstm(G.graphs, h0, c0)
        # end_lstm_time = time.time()

        # construct dp = [↑hpA; ↓hp1; ↓hp2]
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

        # Indexing sequential lstm hidden state using two lists of
        # start and end ids of each unit
        # from list(tuple(start_p1, start_p2)) to two separate lists
        # NOTE We have used the function to split the list of list of tuples
        # `split_nested_list` four times.
        p1_st, p2_st = self.split_nested_list(pair_data["start"], self.device)
        p1_end, p2_end = self.split_nested_list(pair_data["end"], self.device)
        p1_s = range_3d_tensor_index(token_embs,
                                     start=p1_st,
                                     end=p1_end,
                                     pair_batch_num=pair_data['lengths'],
                                     reduce_="mean")
        p2_s = range_3d_tensor_index(token_embs,
                                     start=p2_st,
                                     end=p2_end,
                                     pair_batch_num=pair_data['lengths'],
                                     reduce_="mean")

        # combine embeddings from lstm and from unit embeddings
        # dp´ = [dp; s1,s2]
        pair_embs = torch.cat((tree_pair_embs, p1_s, p2_s), dim=-1)
        # pair_data["graph_time"] = end_graph_time - start_graph_time
        # pair_data["lstm_time"] = end_lstm_time - start_lstm_time
        return pair_embs, pair_data


    def get_pair_preds(
            self, 
            logits : Tensor,
            pair_data : DefaultDict,
            ) -> Tuple[Tensor, Dict[str, Dict[str, Tensor]]]:

        # Get the max among the forward and rev direction
        #ll_preds_prob, ll_preds_id = torch.max(logits, dim=-1)
        ll_logits, ll_preds = torch.max(logits, dim=-1)

        # Get link prediction:
        # Link prediction = the id of the second unit in the pair (u2) if
        # link_label prediction:
        #    1- is not None, otherwise Link prediction = selfloop
        #    2- is not in rev, otherwise swap (u1) and (u2)
        p1 = torch.cat(pair_data["p1"])
        p2 = torch.cat(pair_data["p2"])  # 2nd item in the pair (the
        # possible link prediction)

        # Also start and end indices of pairs' tokens are affected
        p1_st, p2_st = self.split_nested_list(pair_data["start"], p1.device)
        p1_end, p2_end = self.split_nested_list(pair_data["end"], p1.device)

        # NOTE It is better to have abstarction for link label none
        ll_preds_none = ll_preds == 0  # Link label prediction = none
        p2[ll_preds_none] = p1[ll_preds_none]
        p2_st[ll_preds_none] = p1_st[ll_preds_none]
        p2_end[ll_preds_none] = p1_end[ll_preds_none]

        # swap pairs if link label predection is rev
        ll_preds_rev = ll_preds > (self.ll_num - 1)
        p2, p1 = self.swap_elements(p2, p1, ll_preds_rev)
        p2_st, p1_st = self.swap_elements(p2_st, p1_st, ll_preds_rev)
        p2_end, p1_end = self.swap_elements(p2_end, p1_end, ll_preds_rev)

        # I assume that you only have the forward direction f\labels or the
        # link label ids. So, for the pairs that have reverse direction, as we
        # have swaped the p1 and p2, the prediction label will be
        # pred_label_rev - (k - 1), where k is number of label link labels
        # including None. No change in the logits as those will be feeded to
        # the loss function. Prediction may be used in intermediant status
        # visualization or metrics calc.
        ll_preds[ll_preds_rev] -= (self.ll_num - 1)

        pair_link_preds_data = {
            "p1": {
                "id": p1,
                "start": p1_st,
                "end": p1_end
            },
            "p2": {
                "id": p2,
                "start": p2_st,
                "end": p2_end
            },
            "num_pairs": pair_data['lengths']
        }

        return ll_logits, ll_preds, pair_link_preds_data


    def get_ll_targets(self, 
                        targets: Tensor,
                        pair_data: DefaultDict
                       ) -> Tuple[Tensor]:

        #ll_target = token_label['link_label_targets']
        batch_size = targets.size(0)

        # Form the ground truth based on units combinatory
        p1_end, p2_end = self.split_nested_list(pair_data["end"], self.device)

        num_pair = pair_data['lengths']
        idx_0 = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device), num_pair)
        targets = targets[idx_0, p1_end - 1]

        return targets


    def get_negative_relations(self, 
                                token_preds: Tensor,
                                token_targets: Tensor,
                                pair_data: DefaultDict
                               ) -> Tensor:
        # Negative relations
        #seg_preds = token_label['preds']
        #seg_targets = token_label['targets']
        seg_preds = token_preds
        seg_targets = token_targets

        batch_size = seg_preds.size(0)
        seg_targets_pad = seg_targets == -1
        seg_preds[seg_targets_pad] = -1  # set pad token to -1 in predections
        seg_preds_wrong = seg_preds != seg_targets

        # Construct start and end indices for all tokens belongs to each pair
        p1_st, p2_st = self.split_nested_list(pair_data["start"], self.device)
        p1_end, p2_end = self.split_nested_list(pair_data["end"], self.device)

        num_pair = pair_data['lengths']
        idx_0 = torch.repeat_interleave(
            torch.arange(batch_size, device=p1_st.device), num_pair)

        # Get the prediction status (predection == target) for each token in
        # each pair. All tokens need to be predicted correctly
        seg_preds_wrong_p1 = self.split_2dtensor_start_end(
            seg_preds_wrong, p1_st, p1_end, idx_0)
        seg_preds_wrong_p2 = self.split_2dtensor_start_end(
            seg_preds_wrong, p2_st, p2_end, idx_0)

        seg_preds_wrong_p1 = torch.stack(
            [torch.all(pair_seg) for pair_seg in seg_preds_wrong_p1])
        seg_preds_wrong_p2 = torch.stack(
            [torch.all(pair_seg) for pair_seg in seg_preds_wrong_p2])

        negative_relation_pair = torch.logical_and(seg_preds_wrong_p1,
                                                   seg_preds_wrong_p2)

        return negative_relation_pair


    def get_tokens_preds(self, 
                        ll_preds: Tensor, 
                        ll_preds_probs: Tensor,
                        token_mask: Tensor,
                        pair_link_preds_data: DefaultDict
                        ) -> Tuple[Tensor]:
        # Get the maximum probability for each first unit in the pair
        # For pair {(0, 1), (0, 2), ..., (0, p_n)} where n is the number of
        # pairs candidiate that have unit_0 as first candidiate, the link_label
        # prediction is one whil has the maximum probability. The second
        # candidate in that pair (that has the max prob) is the link prediction.

        # NOTE instead of using for loops, I think using dataframe is the most
        # suitable chioce. This is the most convenient straight forward way and
        # arguabily fastest way.
        batch_size = token_mask.size(0)
        num_pairs = pair_link_preds_data["num_pairs"]
        batch_id = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device), num_pairs)
        ll_dat_df = {
            "batch_id": batch_id.tolist(),
            "p1": pair_link_preds_data["p1"]["id"].tolist(),
            "p2": pair_link_preds_data["p2"]["id"].tolist(),
            "probs": ll_preds_probs.tolist(),
            "ll_preds": ll_preds_id.tolist(),
            "p1_start": pair_link_preds_data["p1"]["start"].tolist(),
            "p1_end": pair_link_preds_data["p1"]["end"].tolist(),
            "p2_start": pair_link_preds_data["p2"]["start"].tolist(),
            "p2_end": pair_link_preds_data["p2"]["end"].tolist()
        }

        ll_df = pd.DataFrame(ll_dat_df)
        ll_df.index.name = "serial"
        # get the heighest prob for each p1 in each batch
        ll_df = ll_df.sort_values("probs").drop_duplicates(["batch_id", "p1"],
                                                           keep='last')
        ll_df.sort_index(inplace=True)  # get the order of pairs back

        # fill link prediction tensor
        idx_0 = ll_df.batch_id.values
        span_len = (ll_df.p1_end - ll_df.p1_start).values
        idx_0 = np.repeat(idx_0, span_len)
        idx_1 = np.hstack(list(map(np.arange, ll_df.p1_start, ll_df.p1_end)))

        l_preds = torch.zeros_like(token_mask, dtype=torch.long)
        l_preds_values = np.repeat(ll_df.p2.values, span_len)
        l_preds[idx_0, idx_1] = torch.tensor(l_preds_values,
                                             device=self.device,
                                             dtype=torch.long)
        l_preds[~token_mask] = -1

        ll_preds = torch.zeros_like(token_mask, dtype=torch.long)
        ll_preds_values = np.repeat(ll_df.ll_preds.values, span_len)
        ll_preds[idx_0, idx_1] = torch.tensor(ll_preds_values,
                                              device=self.device,
                                              dtype=torch.long)
        ll_preds[~token_mask] = -1

        return l_preds, ll_preds


    def forward(
                self,
                token_embs: Tensor,
                dep_embs: Tensor,
                one_hot_embs: Tensor,
                roots: Tensor,
                token_mask: Tensor,
                deplinks: Tensor,
                seg_data: dict,
                ):

        self.device = token_embs.device

        # max_units = seg_data["max_units"]

        # essentially, we do 3 things:
        # 1) build a graph
        # 2) pass the graph to lstm to get the dp
        # 3) average token embs to create unit representations
        #
        # we return dp´and the global unit indexes for unit1 and unit2 in pairs

        pair_embs, pair_data = self.build_pair_embs(
                                                    token_embs=token_embs,
                                                    dep_embs=dep_embs,
                                                    one_hot_embs=one_hot_embs,
                                                    roots=roots,
                                                    token_mask=token_mask,
                                                    deplinks=deplinks,
                                                    seg_data=seg_data,
                                                    )
        # We predict link labels for both directions. Get the dominant pair dir
        # plus roots' probabilities
        ll_logits = self.link_label_clf_layer(pair_embs)

        # Get predictions
        ll_logits, ll_preds, pair_link_preds_data = self.get_pair_preds(
                                                                    logits = ll_logits,
                                                                    pair_data = pair_data, 
                                                                    )

        # # Formate predections tensor
        # l_preds_token, ll_pred_tokens = self.get_tokens_preds(
        #                                                         ll_preds_id, 
        #                                                         ll_preds_prob, 
        #                                                         token_mask, 
        #                                                         pair_link_preds_data
        #                                                         )
  
        return logits, ll_preds, l_preds, pair_data




    def loss(self,  targets:Tensor, 
                    logits:Tensor, 
                    token_preds: Tensor,
                    token_targets:Tensor, 
                    pair_data:dict
                    ):

        # Get link label target, based on the predicted pairs
        targets = self.get_ll_targets(    
                                        targets = targets,
                                        pair_data = pair_data
                                        )

    
        # Get negative relations, then set probabilties of ll_prediction to
        # None. Also set prediciton of link label to self loop.
        neg_rel_bool = self.get_negative_relations(
                                                    token_preds = token_preds,
                                                    token_targets = token_targets,
                                                    pair_data = pair_data
                                                    )

  
        loss = F.cross_entropy(
                                torch.flatten(logits,end_dim=-2), 
                                targets.view(-1), 
                                reduction="mean",
                                ignore_index=-1
                                )
        
        return loss




class LinkLabelPairLoss(nn.Module):

    """

    For each pair we create a target label either from the ground truths for that pair if its a true pair, or we 
    treat the target as a negative sample, i.e. the target label will be equivallent to "THIS SHOULD NOT BE PREDICTED TO LINK"-.

    Negative sampling is applied for any pair which satisifies the following conditions:

        1) the label of the members of the pairs are incorrect
        
        2) the pair has no link_label
    
    EXAMPLE:


    """

    def __init__(self, nr_label:int, mode:str):
        self.nr_label = nr_label
        self.mode = mode
        self.loss_fn = CrossEntropyLoss(reduce="mean", ignore_index=-1)


    def forward(
                logits:Tensor, 
                #link_predictions: Tensor,
                #link_targets: Tensor,
                pair_token_idxs: Tensor, 
                token_labels: Tensor, 
                token_targets: Tensor,
                ):
        
        #create a flat list of all candidate pairs
        if self.mode == "pair":
            pair_logits = logits
        else:
            raise NotImplementedError

        #create a target tensor based on the 
  
        # create a mask for selecting pairs which entities are incorrect
        

        # create a mask for selecting pairs where links are incorrect












            
    #def loss(self, targets:dict, logits:dict, seg_data:dict, pair_data:dict):

        # # NOTE it is better to acess the none link_label from init
        # # NOTE should not put the resprctive p1 to be high!
        # ll_preds_id[neg_rel_bool] = 0.0
        # ll_logits_all[neg_rel_bool] = ll_logits_all.new_tensor(
        #     np.log([0.99, 0.0025, 0.0025, 0.0025, 0.0025]))

        # # Set link label prediction to self loop
        # p1 = pair_link_preds_data["p1"]["id"]
        # p2 = pair_link_preds_data["p2"]["id"]
        # p1_st = pair_link_preds_data["p1"]["start"]
        # p2_st = pair_link_preds_data["p2"]["start"]
        # p1_end = pair_link_preds_data["p1"]["end"]
        # p2_end = pair_link_preds_data["p2"]["end"]
        # p2[neg_rel_bool] = p1[neg_rel_bool]
        # p2_st[neg_rel_bool] = p1_st[neg_rel_bool]
        # p2_end[neg_rel_bool] = p1_end[neg_rel_bool]
        # pair_link_preds_data["p2"] = {"id": p2, "start": p2_st, "end": p2_end}

        # # Get link label target, based on the predicted pairs
        # ll_target = self.get_ll_targets(token_label, pair_data)

        # # Formate predections tensor
        # l_preds_token, ll_pred_tokens = self.get_tokens_preds(
        #     ll_preds_id, ll_preds_prob, token_mask, pair_link_preds_data)

        # return {
        #     "link_preds":
        #     l_preds_token,
        #     "link_label_preds":
        #     ll_pred_tokens,
        #     "link_label_logits":
        #     ll_logits_all,
        #     "link_label_target":
        #     ll_target,
        # }