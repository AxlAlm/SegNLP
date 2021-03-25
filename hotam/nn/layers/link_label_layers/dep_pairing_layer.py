        

import torch as torch
import torch.nn as nn

#hotam
from hotam.nn.layers import type_treelstm
from hotam.nn.utils import index_4D


class DepPairingLayer(nn.Module):


    def __init__(self,):
        pass
        self.__supported_modes = set(["shortest_path"])

    def forward(self, 
                input_embs:torch.tensor, 
                dependencies:torch.tensor, 
                pairs:torch.tensor,
                mode:str="shortest_path"
                )
    
        assert mode in self.__supported_modes, f"{mode} is not a supported mode for DepPairingLayer"

        #8) Im unsure how we are going to build the 
        graphs = self.build_dep_graphs(
                                        deplinks = dependencies, 
                                        token_reps = input_embs, 
                                        subgraphs = pairs
                                        )

        #9) 
        #
        tree_lstm_out = self.tree_lstm(graphs)

        #10) Here we should format the data to the following structure:
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
        pairs = ""


        #now we should get logist  for each link_labels
        #(batch_size, nr_units, nr_units, nr_link_labels)
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
        link_label_logits = self.link_label_clf(pairs)

        # 11)
        # first we get the index of the unit each unit links to
        # we do this by first get the highest score of the link label
        # for each unit pair. Then we argmax that to get the index of 
        # the linked unit.
        max_link_label_logits = torch.max(link_label_logits, dim=-1)
        link_preds = torch.argmax(max_link_label_logits, dim=-1)

        # 12)
        # we index the link_label_scores by the link predictions, selecting
        # the logits for the link_labels for the linked pairs
        top_link_label_logits = index_4D(link_label_logits, index=link_preds)


        return top_link_label_logits, link_preds
