# from collections import defaultdict
# from typing import List, Dict, Tuple

import time

from math import exp, floor, factorial
from random import random

import pandas as pd
import torch as th
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph

from hotam.nn.layers.lstm import LSTM_LAYER
from hotam.nn.layers.treelstm import TreeLSTM
from hotam.nn.utils import Graph, pattern2regex, get_all_pairs



# class NELabelEmbedding(nn.Module):
#     def __init__(self, encode_size):

#         super(NELabelEmbedding, self).__init__()
#         self.encode_size = encode_size

#     def forward(self, prediction_id, device):
#         # label prediction, one-hot encoding for label embedding
#         batch_size = prediction_id.size(0)
#         label_one_hot = th.zeros(batch_size, self.encode_size, device=device)
#         label_one_hot[th.arange(batch_size), prediction_id] = 1

#         return label_one_hot


class SegLabeling(nn.Module):
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
        self.seq_clf = nn.Sequential(
                                        nn.Linear(ner_input_size, ner_hidden_size),
                                        nn.Tanh(),
                                        self.dropout,
                                        nn.Linear(ner_hidden_size, ner_output_size),
                                    )

        # # self.label_embs_size = label_embedding_size
        # self.entity_embedding = NELabelEmbedding(
        #     encode_size=label_embedding_size)

    def forward(self,
                word_embs,
                lengths,
                ):

        lstm_out, _ = self.seqLSTM(word_embs, lengths)

        batch_size = lstm_out.size(0)
        max_lenght = lstm_out.size(1)
        hidden_size = lstm_out.size(-1)

        logits = torch.zeros(batch_size, max_lenght, self.label_embedding_size)
        probs = torch.zeros(batch_size, max_lenght, self.label_embedding_size)
        preds = torch.zeros(batch_size, max_lenght)

        word_label_embs =  torch.zeros(batch_size, max_lenght, hidden_size+self.label_embedding_size)

        for i in range(seq_length): 
            
            one_hots = F.one_hot(preds[:,i-1], num_classes=self.label_embedding_size)
            x = th.cat((lstm_out[:, i], one_hot), dim=-1)

            word_label_embs[:, i] = x

            logit = self.seq_clf(x)  # (B, NE-OUT)
            logits[i: ] = logit    
            probs[i: ] = F.softmax(logit, dim=1)  
            preds[i: ] = F.argmax(probs[i: ], dim=1)  

        return logits, probs, preds, lstm_out




class LSTM_ER(nn.Module):
    def __init__(self, hyperparamaters: dict, task2dim: dict, feature2dim: dict):
        super(LSTM_ER, self).__init__()

        # number of arguemnt components
        self.num_ac = len(task2labels["seg_ac"])
        self.num_relations = len(task2labels["stance"])  # number of relations
        self.no_stance = task2labels["stance"].index("None")

        self.p_regex = pattern2regex(task2labels["seg_ac"])

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
        label_embs_size = self.num_ac
        # Embed dimension for dependency labels
        dep_embs_size = feature2dim["deprel_embs"]
        # Sequential LSTM hidden size
        seq_lstm_h_size = hyperparamaters["seq_lstm_h_size"]
        # Tree LSTM hidden size
        self.tree_lstm_h_size = hyperparamaters["tree_lstm_h_size"]
        # Entity recognition layer hidden size
        ac_seg_hidden_size = hyperparamaters["ac_seg_hidden_size"]
        # Entity recognition layer output size
        ac_seg_output_size = self.num_ac
        # Relation extraction layer hidden size
        re_hidden_size = hyperparamaters["re_hidden_size"]
        # Relation extraction layer output size
        re_output_size = self.num_relations
        # Sequential LSTM number of layer
        seq_lstm_num_layers = hyperparamaters["seq_lstm_num_layers"]
        # Sequential LSTM bidirection
        lstm_bidirectional = hyperparamaters["lstm_bidirectional"]
        # Tree LSTM bidirection
        tree_bidirectional = hyperparamaters["tree_bidirectional"]

        dropout = hyperparamaters["dropout"]
        self.dropout = nn.Dropout(dropout)

        self.label_one_hot = NELabelEmbedding(label_embs_size)

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

        # Argument COmponents Segmentation module
        # ------------------------------------------
        self.module_ac_seg = AC_Seg_Module(
            token_embedding_size=token_embs_size,
            label_embedding_size=label_embs_size,
            h_size=seq_lstm_h_size,
            ner_hidden_size=ac_seg_hidden_size,
            ner_output_size=ac_seg_output_size,
            bidirectional=lstm_bidirectional,
            num_layers=seq_lstm_num_layers,
            dropout=dropout)

        # Relation extraction module
        # -----------------------------
        nt = 3 if tree_bidirectional else 1
        ns = 2 if lstm_bidirectional else 1
        re_input_size = self.tree_lstm_h_size * nt + 2 * seq_lstm_h_size * ns
        tree_input_size = seq_lstm_h_size * ns + dep_embs_size + label_embs_size
        self.tree_lstm = TreeLSTM(embedding_dim=tree_input_size,
                                  h_size=self.tree_lstm_h_size,
                                  dropout=dropout,
                                  bidirectional=tree_bidirectional)
        self.rel_decoder = nn.Sequential(
                                        nn.Linear(re_input_size, re_hidden_size), nn.Tanh(), self.dropout,
                                        nn.Linear(re_hidden_size, re_output_size))

    @classmethod
    def name(self):
        return "LSTM_ER"


    def build_graphs(self):

        # Build a batch graph:
        # --------------------------
        graphs = Graph(V=token_head,
                       lengthes=lengths_sent_tok,
                       pad_mask=pad_mask,
                       roots_id=sents_root,
                       graph_buid_type=self.graph_buid_type)
        nodes_input = th.cat((h_seg, dep_embs, pred_embs_seg), dim=-1)
        # update graph data
        graphs.update_batch(ndata_dict={"emb": nodes_input})




        # NOTE AXEL:
        # 1 ) It seems like you go through a lot of trouble getting all pairs and then fitting them into a graph
        #      
        # 2 ) maybe move the graph constructing to the TreeLSTM? E.g. we only need one loop, it will be eaiser to see. We could even construct pass imput batch wise
        # and allow the TreeLSTM to select the spans based on masks?
        #
        # for example:
        #       
        #       for pair in pairs:
        #           graph = create_graph(pairs)       
        #           TreeLSTM(graph)
        
        # Get all possible pairs between the last token of the used argument
        # segments (detected or ground truth)
        relations_data = get_all_pairs(seg_ac_used, pad_mask, self.p_regex)
        # NOTE loop is needed because the sentence lengthes differ across
        rel_graphs = []
        h_ac_dash = []
        rel_ground_truth = []
        stance_ground_truth = []
        ac_seg_neg = []
        batch_id = []
        ac1_2d = []
        ac2_2d = []
        ac1_span_2d = []
        for rel_data in relations_data:
            # no relations, build empty graphs
            (r1, r2) = rel_data[3]
            if r1.nelement() == 0:
                continue

            sample_id = rel_data[0]  # batch number
            (ac1_span, ac2_span) = rel_data[2]  # Index range for AC
            (ac1_id, ac2_id) = rel_data[1]  # AC index in the sample

            ac_map_dict = rel_data[4]
            ac_num = len(ac_map_dict)

             
            # Prepare negative relation data
            # -------------------------------
            # Our relation candidates = all possible combinations of the
            # segmented components
            # The relations canidiates pairs are in following forms:
            #   - last token pairs:     (r1, r2)
            #   - AC id pairs:          (ac1_id, ac2_id)
            #   - AC's tokens id span:  (ac1_span, ac2_span)
            #   Combine relation candidate pairs to account for the two
            #   directions. They are calculated by combinations which account
            #   for one direction only; combination without repeatetion.
            # the second element in the pair: AC2 ids
            ac1_2d_temp = [0] * len(ac1_id) * 2
            ac1_2d_temp[slice(0, len(ac1_id) * 2 + 1, 2)] = ac1_id
            ac1_2d_temp[slice(1, len(ac1_id) * 2 + 1, 2)] = ac2_id

            ac2_2d_temp = [0] * len(ac1_id) * 2
            ac2_2d_temp[slice(0, len(ac1_id) * 2 + 1, 2)] = ac2_id
            ac2_2d_temp[slice(1, len(ac1_id) * 2 + 1, 2)] = ac1_id

            # same for r1, last token for AC1
            r1_2d = [0] * len(ac1_id) * 2
            r1_2d[slice(0, len(ac1_id) * 2 + 1, 2)] = r1 - 1
            r1_2d[slice(1, len(ac1_id) * 2 + 1, 2)] = r2 - 1
            # same for AC1 and AC2 ranges
            ac1_span_2d_temp = [0] * len(ac1_id) * 2
            ac1_span_2d_temp[slice(0, len(ac1_id) * 2 + 1, 2)] = ac1_span
            ac1_span_2d_temp[slice(1, len(ac1_id) * 2 + 1, 2)] = ac2_span

            ac2_span_2d = [0] * len(ac1_id) * 2
            ac2_span_2d[slice(0, len(ac1_id) * 2 + 1, 2)] = ac2_span
            ac2_span_2d[slice(1, len(ac1_id) * 2 + 1, 2)] = ac1_span

            # A pair will have a negative relation when the argument
            # segmentation is wrong or when the pair has no relation.
            # Using end_token_i (first candidate) ==get==> the ground truth of
            # AC_id_gth
            # Compare extracted AC_id_gth with the AC_id_j (second candidate)
            rel_ground_truth_i = batch['relation'][sample_id][r1_2d]
            # only pairs that have relations
            rel_truth_bool_i = rel_ground_truth_i == th.tensor(ac2_2d_temp)
            rel_ground_truth.extend(rel_truth_bool_i)

            # check that the number of extracted rel_ground_truth is equal to
            # the number of argument components we have.
            rel_root_num = 0
            for last_token_id, ac_id in ac_map_dict.items():
                if ac_id == batch['relation'][sample_id][last_token_id - 1]:
                    rel_root_num += 1
            assert rel_truth_bool_i.sum() == ac_num - rel_root_num

            # get AC segmentation ground truth and predection using AC1 range
            ac_seg_truth = batch["seg_ac"][sample_id]  # AC segmentation truth
            ac_seg_predict = seg_ac_used[sample_id]  # AC segs prediction
            ac1_neg_status = th.stack([
                ~th.all(ac_seg_truth[i] == ac_seg_predict[i])
                for i in ac1_span_2d_temp
            ])  # AC1 segmentation is wrong
            ac2_neg_status = th.stack([
                ~th.all(ac_seg_truth[i] == ac_seg_predict[i])
                for i in ac2_span_2d
            ])  # AC2 segmentation is wrong
            # either AC1 or AC2 segmentation is wrong
            ac_seg_neg.extend(th.bitwise_or(ac1_neg_status, ac2_neg_status))

            # get stance ground truth
            stance_truth_i = batch['stance'][sample_id][r1_2d]
            # set stance of pairs that have no relation to be 0
            stance_truth_i[~rel_truth_bool_i] = self.no_stance
            stance_ground_truth.extend(stance_truth_i)

            # get some data to be used to construct the predictions
            batch_id.extend([sample_id] * len(r1_2d))
            ac1_2d.extend(ac1_2d_temp)
            ac2_2d.extend(ac2_2d_temp)
            ac1_span_2d.extend([span.tolist() for span in ac1_span_2d_temp])

            # Creat subtrees
            # --------------
            s_i = h_seg[sample_id].detach().cpu()
            sub_graphs, h_dash = graphs.get_subgraph_data(batch_id=sample_id,
                                                          starts=r1,
                                                          ends=r2 - 1,
                                                          ac1_idx=ac1_span,
                                                          ac2_idx=ac2_span,
                                                          h=s_i)
            rel_graphs.extend(sub_graphs)
            h_ac_dash.append(h_dash)



    def forward(self, batch):

        device = self.model_param.device

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


        # # REVIEW sometime_s I recieved lengthes in tensor!
        # if type(lengths_sent_tok) == Tensor:
        #     lengths_sent_tok = lengths_sent_tok.tolist()

        # sents_root = batch["sent2root"]  # Tensor[B, SENT_NUM]
        # if (sents_root.size(1) >= token_head.size(1)):
        #     _SENT_NUM_ = token_head.size(1)
        #     sents_root = sents_root[:, :_SENT_NUM_]

        # batch_size = token_embs.size(0)

    
        input_ac_seg = th.cat((token_embs, pos_embs), dim=2)

        word_embs = self.dropout(word_embs)

        logits, probs, preds, word_label_embs = self.module_ac_seg(word_embs, lengths_per_sample)


        nodes_input = th.cat((word_label_embs, dep_embs), dim=-1)


        # Get relations
        # --------------
        # all possible  relations in both directions between the last tokens
        # of the detected entities.
        schdule_sampling = self.k / (self.k +
                                     exp(batch.current_epoch / self.k))
        # schdule sampling
        coin_flip = floor(random() * 10) / 10
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
        # ======================================================================




        if rel_graphs:
            rel_graphs = dgl.batch(rel_graphs).to(device)  # type: DGLGraph
            h_ac_dash = th.cat(tuple(h_ac_dash), dim=0).to(device)
            # =============================================================

            # Relation extraction module:
            # ---------------------------
            h0 = th.zeros(rel_graphs.num_nodes(),
                          self.tree_lstm_h_size,
                          device=device)
            c0 = th.zeros_like(h0)

            tree_rep = self.tree_lstm(rel_graphs, h0, c0, h_ac_dash)
            stance_logits = self.rel_decoder(tree_rep)

            stance_prob_dist = F.softmax(stance_logits, dim=1)
            stance_prob, stance_predict = th.max(stance_prob_dist, dim=1)

            # REVIEW how neg rel effect the loss calc ?! here it does not
            # Negative relations:
            # =================================================================
            rel_ground_truth = th.stack(rel_ground_truth)
            # if stance > 0, then rel is predicted
            rel_predict_bool = stance_predict > self.no_stance
            rel_neg = (rel_predict_bool != rel_ground_truth)
            neg_id = th.bitwise_or(rel_neg, th.stack(ac_seg_neg)).type(
                th.int) * -1

            rel_predict = th.tensor(ac2_2d) * rel_predict_bool * neg_id
            # =================================================================

            # REVIEW this part below is only needed to form stance and pred
            # tensor as required to have size of B, SEQ
            # get relations and stance predictions in B, SEQ
            data = {
                "batch_id": batch_id,
                "ac_id": ac1_2d,
                "rel_p": rel_predict.tolist(),
                "stance_p": stance_predict.tolist(),
                "stance_pb": stance_prob.tolist(),
                "ac_span": ac1_span_2d
            }
            rel_stance_df = pd.DataFrame(data)
            rel_stance_df.index.name = "serial"
            # get max stance prob. for each ac_id for each batch
            # we have all possible pairs relations:
            # For batch x
            #   ac_id   , rel_p , stance_p , stance_pb |    Pair
            # -----------------------------------------|  ========
            #     1     ,   0   ,    0     ,    0.2    |  AC1, AC2
            #     2     ,   1   ,    4     ,    0.5    |  AC2, AC1
            #     1     ,   0   ,    0     ,    0.3    |  AC1, AC3 **
            #     3     ,   2   ,    5     ,    0.4    |  AC3, AC1
            #     2     ,   0   ,    0     ,    0.4    |  AC2, AC3
            #     3     ,   1   ,    4     ,    0.3    |  AC3, AC2
            #
            # so for AC_1 we choose pair (AC1, AC3) as they have the highest
            # prob max(0.3, 0.2). We do this for each AC in each batch.

            # get the heighest prob for each AC in each batch
            rel_stance_df = rel_stance_df.sort_values(
                'stance_pb').drop_duplicates(['batch_id', 'ac_id'],
                                             keep='last')

            rel_stance_df.sort_index(
                inplace=True)  # get the order of pairs back
            pred_rel = th.zeros_like(pred_seg)
            pred_stance = th.zeros_like(pred_seg)

            df = rel_stance_df.explode('ac_span')  # expand AC indices to rows
            # fill relation and stance predictions tensors using indices from df
            pred_rel[df.batch_id.to_list(),
                     df.ac_span.to_list()] = th.tensor(df.rel_p.to_list(),
                                                       dtype=th.float)
            pred_stance[df.batch_id.to_list(),
                        df.ac_span.to_list()] = th.tensor(
                            df.stance_p.to_list(), dtype=th.float)

        # Calculation of losses:
        # ----------------------
        # Argument Components Segmentation Module.
        # (B, SEQ, NE-OUT) --> (B*SEQ, NE-OUT)
        logitss_seg = logitss_seg.view(-1, logitss_seg.size(-1))
        batch.change_pad_value(-1)  # ignore -1 in the loss function
        ac_seg_ground_truth = batch["seg_ac"].view(-1)
        loss_seg = self.loss(logitss_seg, ac_seg_ground_truth)

        # Relation extraction:
        # --------------------
        if rel_graphs:
            stance_ground_truth = th.stack(stance_ground_truth)
            loss_stance = self.loss(stance_logits, stance_ground_truth)

        else:  # no relation decovered
            # REVIEW calculation of logits an loss when no stance is predicted
            f = factorial
            rel = self.num_relations
            num_pairs = int(f(rel) // f(2) // f(rel - 2))
            stance_logits = th.zeros(num_pairs,
                                     rel,
                                     dtype=th.float,
                                     requires_grad=True,
                                     device=device)
            stance_logits[:, self.no_stance] = 100
            # REVIEW does not matter the value of stance  groung truth!
            stance_ground_truth = th.ones(num_pairs,
                                          dtype=th.long,
                                          device=device)
            loss_stance = self.loss(stance_logits, stance_ground_truth)

            pred_stance = th.ones_like(batch["stance"]) * -2
            pred_rel = th.ones_like(batch["stance"]) * -2

        loss_total = loss_seg + loss_stance

        end_time = time.time()
        hr, mins, secs = calculate_time(start_time, end_time)
        # print(f"\nProcess data finished. ET:{hr:2d}:{mins:2d}:{secs:2d}\n")

        return {
            "loss": {
                "total": loss_total,
            },
            "preds": {
                "seg_ac": pred_seg,
                "relation": pred_rel,
                "stance": pred_stance
            },
            # REVIEW calculation, size required for stance prob
            "probs": {
                "seg_ac": prob_seg,
                # "relation": relation_probs,
                # "stance": stance_prob
            },
        }
