from typing import List, Tuple, Dict

from collections import defaultdict
import itertools as it
from operator import itemgetter
import re

import torch
from torch import Tensor
import numpy as np

import networkx as nx
from networkx import Graph as nxGraph
import dgl
from dgl import DGLGraph
from dgl.traversal import topological_nodes_generator as traverse_topo


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
        g_list = []
        device = torch.device("cpu")

        V = V.detach().cpu()
        pad_mask = pad_mask.detach().cpu()
        roots_id = roots_id.detach().cpu()
        for i, data in enumerate(zip(lengthes, V, roots_id, pad_mask)):
            lens_, v_i, root_id, p_mask = data
            lens = torch.tensor(lens_ + [-1] * (v_i.size(0) - len(lens_)))
            mask = lens[:, None] > torch.arange(v_i.size(1))
            # token_ids of the depdency heads are w.r.t. sentence. So, in v_i
            # token_id = 1, could be in sentence_1 or sentence_2. However, we
            # deal with ids w.r.t. sample, meaning that id sentence_0 has 5
            # tokens, then token_id=0 in sentence_1 will be 6
            # Calculate Acculmulative sum for length, to get absolute ids
            lens_accum = torch.cumsum(torch.tensor(lens), dim=0)
            delta2abs = lens_accum[:-1]  # (SNT_N - 1)
            v_i_abs = v_i + torch.cat((torch.tensor([0]), delta2abs))[:, None]

            v = torch.masked_select(v_i_abs, mask=mask)  # select non pad token
            u = torch.arange(torch.squeeze(mask).sum(),
                             dtype=torch.long,
                             device=device)

            # convert root_id to absolute ids w.r.t sample just as we did with
            # v_i
            roots_abs = root_id + torch.cat((torch.tensor([0]), delta2abs))
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

            g_list.append(g)
            self.device = device

        # self.G = dgl.batch(all_graph)
        self.mask = pad_mask  # (B, n_tokens)
        self.G = g_list
        # self.nonpad_idx = torch.squeeze(torch.nonzero(pad_mask.view(-1)))
        # self.batch_update_nodes(ndata=ndata)

    def connect_sents(self, u: Tensor, v: Tensor, roots: Tensor,
                      lengths_accum: Tensor,
                      graph_buid_type: int) -> Tuple[Tensor, Tensor, Tensor]:
        if graph_buid_type == 0:
            # connect the root of the current sent to the end of the prev sent
            v[roots[1:]] = lengths_accum[:-1] - 1
            # remove the self connection of the first root
            u = torch.cat((u[:roots[0]], u[roots[0] + 1:]))
            v = torch.cat((v[:roots[0]], v[roots[0] + 1:]))
            # self.token_rmvd.append(roots[0].item())
            roots = roots[1:]
        return u, v, roots

    def __extract_graph(self, g: DGLGraph, g_nx: nxGraph, start_n: list,
                        end_n: list, sub_graph_type: int):
        """
        """
        if sub_graph_type == 0:
            thepath = nx.shortest_path(g_nx, source=start_n, target=end_n)
            sub_g = dgl.node_subgraph(g, thepath)
            root = list(traverse_topo(sub_g))[-1]
            assert len(root) == 1
            sub_g.ndata["type_n"] = torch.zeros(sub_g.number_of_nodes(),
                                                device=self.device,
                                                dtype=torch.long)
            str_mark = torch.zeros(sub_g.number_of_nodes(), device=self.device)
            end_mark = torch.zeros_like(str_mark)
            root_mark = torch.zeros_like(str_mark)

            root_mark[root] = 1
            sub_g.ndata["root"] = root_mark

            str_mark[0] = 1
            end_mark[-1] = 1
            sub_g.ndata["start"] = str_mark
            sub_g.ndata["end"] = end_mark

            # check ...
            assert sub_g.ndata["_ID"][0] == start_n
            assert sub_g.ndata["_ID"][-1] == end_n
            assert str_mark.sum() == end_mark.sum()
            assert str_mark.sum() == root_mark.sum()

        elif sub_graph_type == 1:
            # get subtree
            pass
        return sub_g

    def get_subgraph_data(
        self,
        batch_id: int,
        starts: Tensor,
        ends: Tensor,
        ac1_idx: List[Tensor],
        ac2_idx: List[Tensor],
        h: Tensor,
        sub_graph_type: int = 0,
    ):
        """
        """
        g = self.G[batch_id]
        g_nx = nx.Graph(g.to_networkx().to_undirected())
        graphs_sub = []
        h_dash = []
        for i in range(starts.size(0)):
            ac1_ids, ac2_ids = ac1_idx[i], ac2_idx[i]
            ac1_h, ac2_h = h[ac1_ids, :], h[ac2_ids, :]
            h_dash.append(torch.cat((ac1_h.mean(0), ac2_h.mean(0)), dim=-1))
            h_dash.append(torch.cat((ac2_h.mean(0), ac1_h.mean(0)), dim=-1))

            src, dst = starts[i].tolist(), ends[i].tolist()
            g_sub = self.__extract_graph(g, g_nx, src, dst, sub_graph_type)
            g_sub_r = self.__extract_graph(g, g_nx, dst, src, sub_graph_type)
            graphs_sub.extend([g_sub, g_sub_r])

        return graphs_sub, torch.stack(h_dash)

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


def pattern2regex(labels):
    """
    """
    #
    # 14*|25*|36*|4+|5+|6+
    labels_s = sorted(enumerate(labels), key=itemgetter(1))
    label_data = defaultdict(list)
    for (idx, label) in labels_s:
        label_data[label[0]].append(idx)

    BI = zip(label_data["B"], label_data["I"])
    BI_astrik = [rf"{b}{i}*" for b, i in BI]
    I_plus = [rf"{i}+" for i in label_data["I"]]
    pattern_regx = re.compile("|".join(BI_astrik + I_plus))

    return pattern_regx


def get_all_pairs(predictions, token_mask, pattern_r):
    """
    """

    device = torch.device('cpu')
    batch_num = predictions.size(0)
    # get all possible pairs of the last token ids
    for i in range(batch_num):
        prdkxn = predictions[i]
        mask = token_mask[i]
        # Select predictions for non pad tokens
        # Convert the selected prediction into string, then use regex to select
        # the argument components ande get their start and end charachters and
        # len
        prdkxn_str = "".join(map(str, prdkxn[mask].type(torch.int).tolist()))
        match = re.finditer(pattern_r, prdkxn_str)
        span = list(
            map(lambda x: (x.start(), x.end(), x.end() - x.start()),
                list(match)))
        ac_start_id, ac_end_id, ac_len = list(zip(*span))
        # map AC last token id to ac_id
        ac_end_token2id = dict(zip(ac_end_id, range(len(ac_end_id))))

        # get all possible pair combinations of the AC last token
        end_r = list(map(list, zip(*it.combinations(ac_end_id, 2))))
        end_r1 = torch.tensor(end_r[0], device=device)  # 1st item in the pair
        end_r2 = torch.tensor(end_r[1], device=device)  # 2nd item in the pair

        #  pair in term of: AC id
        ac1_id = [ac_end_token2id[r1.item() - 1] for r1 in end_r1]
        ac2_id = [ac_end_token2id[r2.item() - 1] for r2 in end_r2]

        # pair in term of: range(AC_start : AC_end)
        strt_r = list(map(list, zip(*it.combinations(ac_start_id, 2))))
        strt_r1 = torch.tensor(strt_r[0], device=torch.device('cpu'))
        strt_r2 = torch.tensor(strt_r[1], device=torch.device('cpu'))
        ac1_r_idx = list(map(torch.arange, strt_r1, end_r1))
        ac2_r_idx = list(map(torch.arange, strt_r2, end_r2))

        yield i, (ac1_id, ac2_id), (ac1_r_idx,
                                    ac2_r_idx), (end_r1,
                                                 end_r2), ac_end_token2id


def masked_mean(m, mask):
    """
    means the rows in a given matrix based on the masked values.

    example:

    if v == m[1] and mask[i] == [0,1,1,1,0]
    mask_mean(v) = mean(v[1:4])

    Parameters
    ----------
    m : tensor
        matrix to be averaged
    mask : mask
        matrix of where m[i,j] = [0,1]

    Returns
    -------

        [description]
    """
    m_sum = torch.sum(m, dim=2)
    mask_sum = torch.sum(mask, dim=2, keepdim=True)
    masked_mean = m_sum / mask_sum
    masked_mean[masked_mean != masked_mean] = 0.0
    return masked_mean


def multiply_mask_matrix(matrix, mask):
    """
        Tedious way of multiplying a 4D matrix with a 3D mask.

        example when we need to do this is when we have a matrix of word
        embeddings
        for paragraphs  (batch_size, nr_paragraphs, nr_tok, word_emb) and we
        want
        to get spans of the paragraphs based on a mask.

        We can then use the mask which signify the spans to turn everything we
        dont
        want to zeros
    """
    og_shape = matrix.shape

    # flatten to 2D
    matrix_f = torch.flatten(matrix, end_dim=-2)

    # 2 flatten mask to 2D, last value needs to be [0] or [1]
    mask_f = mask.view((np.prod([matrix.shape[:-1]]), 1))

    # 3 turn embs into 0 where mask is 0
    matrix_f_masked = matrix_f * mask_f

    # 4 return to original shape
    masked_matrix = matrix_f_masked.view(og_shape)

    return masked_matrix


def reduce_and_remove(matrix, mask):
    """
    Given a 4D matrix turn it into a 3D by removing 3D dimension while
    perserving padding.

    (similar to pytroch.utils.pad_packed_sequences, sort of)


    for example:
        given a 4D matrix where dims are (batch_size, nr_paragraphs, nr_spans,
        nr_tokens),
        if we want to get all words for all paragraphs we need to remove spans
        and remove padding tokens.
        we cannot remove all values == n as padding for nr words in paragraphs
        needs to exists.
        So, we need to find out max paragraph length, remove all zeros between
        that length and then after.

    """
    batch_size, _, _, feature_dim = matrix.shape

    matrix_f = matrix[mask]
    lengths = torch.sum(torch.sum(mask, dim=-1), dim=1)

    new_tensor = torch.zeros((batch_size, max(lengths), feature_dim))

    start_idx = 0
    for end_idx in lengths:
        new_tensor[:end_idx] = matrix_f[start_idx:start_idx + end_idx]
        start_idx += end_idx

    return new_tensor
