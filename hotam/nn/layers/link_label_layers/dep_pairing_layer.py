
#basics
from typing import List, Tuple, DefaultDict
import functools
import itertools
import numpy as np 

#pytorch
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# networkx
import networkx as nx
from networkx import Graph as nxGraph

#DGL
import dgl
from dgl import DGLGraph
from dgl.traversal import topological_nodes_generator as traverse_topo

# hotam
from hotam.nn.model_input import ModelInput
from hotam.nn.layers.type_treelstm import TypeTreeLSTM
from hotam.nn.utils import scatter_repeat
from hotam.nn.utils import get_all_possible_pairs
from hotam.nn.utils import pair_matrix
from hotam.nn.utils import agg_emb
from hotam.nn.utils import create_mask
from hotam.nn.utils import index_select_array
from hotam.nn.utils import cumsum_zero



class DepGraph:
    
    def __init__(self, 
                token_embs: Tensor, 
                deplinks: Tensor,
                roots: Tensor, 
                token_mask: Tensor, 
                subgraphs: List[List[Tuple]],
                mode: str,
                device=None,
                assertion:bool=False
                ) -> List[DGLGraph]:

        assert mode in set(["shortest_path"]), f"{mode} is not a supported mode for DepPairingLayer"
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

        pre_clf_layer =  nn.Sequential(
                                            nn.Linear(decoder_input_size, decoder_h_size), 
                                            nn.Tanh(),
                                            nn.Dropout(dropout), 
                                            )
        self.left2right = nn.Sequential(    
                                        pre_clf_layer,
                                        nn.Linear(decoder_h_size, decoder_output_size),
                                        nn.Softmax(dim=-1)
                                        )
        self.right2left = nn.Sequential(
                                        pre_clf_layer,
                                        nn.Linear(decoder_h_size, decoder_output_size),
                                        nn.Softmax(dim=-1)
                                        )                                 


    def build_pair_embs(self,
                        token_embs: Tensor,
                        dep_embs : Tensor,
                        one_hot_embs : Tensor,
                        roots: Tensor,
                        token_mask: Tensor,
                        deplinks: Tensor,
                        bio_data: dict,
                        mode: str = "shortest_path",
                        assertion: bool = False
                        ):

        # get all possible pairs
        pair_data = get_all_possible_pairs(
                                            start=bio_data["unit"]["start"],
                                            end=bio_data["unit"]["end"],
                                            )
                            
        # 1) Build graph from dependecy data
        node_embs = torch.cat((token_embs, one_hot_embs, dep_embs), dim=-1)
        G = DepGraph(
                        token_embs=node_embs,
                        deplinks=deplinks,
                        roots=roots,
                        token_mask=token_mask,
                        subgraphs=pair_data["end"],
                        mode=mode,
                        device=self.device,
                        assertion=assertion
                    )

        # 2) Pass graph to a TreeLSTM to create hidden representations
        h0 = torch.zeros(
                        G.graphs.num_nodes(),
                        self.tree_lstm_h_size,
                        device=self.device
                        )
        c0 = torch.zeros_like(h0)
        tree_lstm_out = self.tree_lstm(G.graphs, h0, c0)

        # construct dp = [↑hpA; ↓hp1; ↓hp2]
        # ↑hpA: hidden state of dep_graphs' root
        # ↓hp1: hidden state of the first token in the candidate pair
        # ↓hp2: hidden state of the second token in the candidate pair
        # get ids of roots and tokens in relation
        root_id = (G.graphs.ndata["root"] == 1)
        start_id = G.graphs.ndata["start"] == 1
        end_id = G.graphs.ndata["end"] == 1

        tree_lstm_out = tree_lstm_out.view(-1, 2, self.tree_lstm_h_size)
        tree_pair_embs = torch.cat((
                                    tree_lstm_out[root_id, 0, :],  # ↑hpA
                                    tree_lstm_out[start_id, 1, :], # ↓hp1
                                    tree_lstm_out[end_id, 1, :]    # ↓hp2
                                    ), 
                                    dim=-1)

        # 3) create unit embeddings
        unit_embs_flat = agg_emb(
                            token_embs, 
                            bio_data["unit"]["lengths"], 
                            bio_data["unit"]["span_idxs"],
                            mode="average",
                            flat=True, #flat will returned all unit embs with padding removed in a 2D tensor
                            ) 
        # p1 and p2 are the unit indexs of each pair.
        # .e.g   p1[k] = token_at_idx_i_in_sample_k,  p2[k] = unit_at_idx_j_in_sample_k
        p1 = torch.hstack(pair_data["p1"])
        p2 = torch.hstack(pair_data["p2"])

        # p1 and p2 are indexes of each pair, but we want to create a flat tensor with all the pairs
        # by selecting them using indexing for a flat tensor of unit_embeddings.
        # hence we need to update the indexes so that each unit index in p1/p2, is relative
        # to the global index of all units
        cum_lens = cumsum_zero(torch.LongTensor(bio_data["unit"]["lengths"]))
        global_index = torch.repeat_interleave(cum_lens, pair_data["lengths"])

        p1g = p1 + global_index
        p2g = p2 + global_index

        unit_pair_embs = torch.cat((unit_embs_flat[p1g], unit_embs_flat[p2g]), dim=-1)


        # combine embeddings from lstm and from unit embeddings
        # dp´ = [dp; s1,s2] 
        pair_embs = torch.cat((tree_pair_embs, unit_pair_embs), dim=-1)


        return pair_embs, p1g, p2g


    def choose_direction(self, left2right_probs:Tensor, rigth2left_probs:Tensor):

        # we take the max prob from both directions
        # maxs = [
        #          [score_L2R, score_RTL]
        #           ...
        #        ]
        maxs = torch.cat((
                                torch.max(left2right_probs,dim=-1)[0].unsqueeze(-1), 
                                torch.max(rigth2left_probs,dim=-1)[0].unsqueeze(-1)
                                ),
                                dim=-1
                                )

        # we max the max of the direction probs to get the dominant direction
        directions = torch.argmax(maxs, dim=-1)

        # we create a global idx over all pairs
        glob_dir_idx = directions + cumsum_zero(torch.full((maxs.shape[0],), 2))

        # essentially perform flatten and zip() to get the probabilities for all paris and 
        # directions so we can index them.
        # e.g. [
        #       pair0_left_to_fight_probs,
        #       pair0_right_to_left_probs,
        #       ...
        #
        #       ]
        #
        cat_dir_probs = torch.cat(( 
                                left2right_probs,
                                rigth2left_probs
                                ),
                                dim=-1
                                ).view(left2right_probs.shape[0],2,3)
        flat_dir_probs = torch.flatten(cat_dir_probs, end_dim=-2)

        #select the probabilties for the best direction
        pair_probs = flat_dir_probs[glob_dir_idx]

        return pair_probs, directions


    def get_outputs(self, 
                    pair_probs:torch.tensor, 
                    p1g:torch.tensor,
                    p2g:torch.tensor,
                    directions:torch.tensor,
                    bio_data:dict,
                    max_units:int, 
                    batch_size:int, 
                    max_tokens:int,
                    token_label:dict=None,
                    ) -> dict:
     
        nr_link_labels = pair_probs.shape[-1]
        max_units = bio_data["max_units"]
        device = pair_probs.device
  
        outputs = {
                    "link_preds": torch.zeros(
                                                (batch_size, max_tokens), 
                                                dtype=torch.long, 
                                                device=device,
                                                ),
                    "link_label_preds": torch.zeros(
                                                    (batch_size, max_tokens), 
                                                    dtype=torch.long, 
                                                    device=device,
                                                    ),
                    # "link_probs": torch.zeros(
                    #                             (batch_size, max_tokens, max_units), 
                    #                             dtype=torch.float,
                    #                             device=device,
                    #                             ),
                    "link_label_probs": torch.zeros(
                                                    (batch_size, max_tokens, nr_link_labels), 
                                                    dtype=torch.float, 
                                                    device=device,
                                                    ),
                    }

        #we set the default prob for all tokens to be link_label "None"
        # value set so not to produce any -inf
        outputs["link_label_probs"][:,:] = torch.FloatTensor([0.999,0.0005,0.0005], device=device)

        # first we figure out which of the members of the pairs 
        # are the roots. Is root at idx 1 or index 0. We can figure this 
        # out with the help of the directions
        where_0_is_root = (directions==0).type(torch.LongTensor)
        where_1_is_root = (directions==1).type(torch.LongTensor)
        roots = (p1g*where_0_is_root) + (p2g*where_1_is_root)

        # COLUMNS: 
        #       index, 
        #       global index of unit 1 in pair, 
        #       global index of unit 2 in pair,
        #       global index of unit 1 OR 2 in pair (e.g. the root)
        pair_info = torch.cat((
                                torch.arange(p1g.shape[0]).unsqueeze(-1), 
                                p1g.unsqueeze(-1), 
                                p2g.unsqueeze(-1), 
                                roots.unsqueeze(-1)
                                ),
                                dim=1
                                )

        # we group the pair info by where the link originates from.
        # i.e. if we have a pair at i pair_info[i] = (i,0,1,0) where 
        #           pair[1] = index of unit 1
        #           pair[2] = index of unit 1
        #           pair[3] = root (index of unit 1 OR 2)
        # then the index of the unit that is at the root is at pair[-1]
        # this means that (i,2,0,0) and (i,0,1,0) are in the same group 
        pair_groups = itertools.groupby(pair_info, lambda x: x[-1])
        

        sample_idx = torch.repeat_interleave(
                                            torch.arange(batch_size), 
                                            repeats=torch.LongTensor(bio_data["unit"]["lengths"]), 
                                            dim=0
                                            )

        rel_unit_idxs = cumsum_zero(torch.LongTensor(bio_data["unit"]["lengths"]))   
        start_token_idxs = torch.LongTensor(np.hstack(bio_data["unit"]["start"]))
        end_token_idxs = torch.LongTensor(np.hstack(bio_data["unit"]["end"]))
        
        for root, group in pair_groups:
            data = torch.stack(list(group))
            idxs = data[:,1]
            ps = data[:,1:3]
            
            #print(root)
            #we get the probabilites for this group, i.e. the proabilites over which some unit is related
            group_link_label_probs = pair_probs[idxs.squeeze(-1)]

            # then we get the max link_label value across rows
            max_link_label_probs = torch.max(group_link_label_probs)

            # then we get the index of pair which is most probable
            max_i = torch.argmax(max_link_label_probs)

            #then we get the prob dist for the link labels (ll)
            link_label_probs = group_link_label_probs[max_i]

            # then we get the label
            link_label = torch.argmax(link_label_probs)

            # first we pick out the unit indexes in the pairs
            # that are not the root
            children = ps[ps != root]

            # pick sample id
            k = sample_idx[root]

            #normalize the linked unit indexes to the sample (global idx -> sample idx)
            linked_unit_idxs = children - rel_unit_idxs[k]

            #pick the predicted linked unit idx
            link = linked_unit_idxs[max_i]

            # normalize the root indx
            root_unit_idx = root - rel_unit_idxs[k]
        

            i = start_token_idxs[root_unit_idx]
            j = end_token_idxs[root_unit_idx]

            n = end_token_idxs[link]

            # if we are given the labels of the units (i.e. "label")
            # We assume that the label of the last token of each unit represent the 
            # label of the whole unit as end tokens represent units throughout the paper
            # and there is not explicit information about voting or some label aggregation method
            # for units
            if token_label is not None:

                p1_not_correct = token_label["preds"][k][j] != token_label["targets"][k][j]
                p2_not_correct = token_label["preds"][k][n] != token_label["targets"][k][n]

                if p1_not_correct and p2_not_correct:
                    link_label_probs = torch.FloatTensor([0.999,0.0005,0.0005], device=device) # value set so not to produce any -inf
                    link_label = 0
                    link = 0
                    #link_probs = ???


            outputs["link_preds"][k][i:j] = link
            outputs["link_label_preds"][k][i:j] = link_label
            outputs["link_label_probs"][k][i:j] = link_label_probs
            #output["link_probs"][k][i:j] = link_label_probs

        return outputs


    def forward(self,
                token_embs: Tensor,
                dep_embs : Tensor,
                one_hot_embs : Tensor,
                roots: Tensor,
                token_mask: Tensor,
                deplinks: Tensor,
                bio_data: dict,
                mode: str = "shortest_path",
                token_label: dict = None,
                assertion: bool = False,
                ):

        self.device = token_embs.device
        batch_size = token_embs.shape[0]
        max_tokens = token_mask.shape[-1]
        max_units = bio_data["max_units"]

        # first want to create the "candidate vector" dp´ for each possible pair
        # dp = [↑hpA; ↓hp1; ↓hp2]
        # dp´= [dp; s1, s2] where s is an unit embedding constructed form average token embs 
        #
        # essentially, we do 3 things:
        # 1) build a graph
        # 2) pass the graph to lstm to get the dp
        # 3) average token embs to create unit representations
        #
        # we return dp´and the global unit indexes for unit1 and unit2 in pairs
        pair_embs, p1g, p2g = self.build_pair_embs(
                                                    token_embs=token_embs,
                                                    dep_embs=dep_embs,
                                                    one_hot_embs=one_hot_embs,
                                                    roots=roots,
                                                    token_mask=token_mask,
                                                    deplinks=deplinks,
                                                    bio_data=bio_data,
                                                    mode=mode,
                                                    assertion=assertion
                                                )        

        # We predict link labels for both directions
        l2r_probs = self.left2right(pair_embs)
        r2l_probs = self.right2left(pair_embs)

        # We get two predictions/"labels" for each pair and then we chose the direction
        # based on the highest link_label value
        # we return the probability distributions of link_labels of dominant directions
        # along with a tensor will directions, e.g. 0 means its left to right  1 the opposite.
        pair_probs, directions = self.choose_direction(
                                                        left2right_probs = l2r_probs,
                                                        rigth2left_probs = r2l_probs,
                                                        )

        outputs = self.get_outputs(
                                    pair_probs=pair_probs, 
                                    p1g = p1g,
                                    p2g = p2g,
                                    directions = directions,
                                    bio_data = bio_data,
                                    max_units = max_tokens, 
                                    batch_size = batch_size, 
                                    max_tokens = max_tokens,
                                    token_label = token_label,
                                    )

                    
        return outputs
