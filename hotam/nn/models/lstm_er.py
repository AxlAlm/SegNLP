

from hotam.nn.layer.seg_layers import BigramSegLayer
from hotam.nn.layers.lstm import LSTM_LAYER
from hotam.nn.utils import index_4D 

class LSTM_ER(torch.nn.Module):

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
        token_embs_size = feature2dim["word_embs"] + feature2dim["pos_embs"]
        label_embs_size = self.num_ac
        dep_embs_size = feature2dim["deprel_embs"]
        seq_lstm_h_size = hyperparamaters["seq_lstm_h_size"]
        self.tree_lstm_h_size = hyperparamaters["tree_lstm_h_size"]
        ac_seg_hidden_size = hyperparamaters["ac_seg_hidden_size"]
        ac_seg_output_size = self.num_ac
        re_hidden_size = hyperparamaters["re_hidden_size"]
        re_output_size = self.num_relations
        seq_lstm_num_layers = hyperparamaters["seq_lstm_num_layers"]
        lstm_bidirectional = hyperparamaters["lstm_bidirectional"]
        tree_bidirectional = hyperparamaters["tree_bidirectional"]

        dropout = hyperparamaters["dropout"]
        self.dropout = nn.Dropout(dropout)

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)


        self.lstm = LSTM_LAYER(
                                input_size=,
                                hidden_size=,
                                num_layers=,
                                bidirectional=,
                                dropout=
                                )

        self.seg_label_clf = BigramSegLayer(
                                                input_size= ,
                                                hidden_size= ,
                                                output_size= ,
                                                label_emb_dim= ,
                                                dropout=dropout,
                                            )   




        self.stanc_link_clf = ""



        # # Relation extraction module
        # # -----------------------------
        # nt = 3 if tree_bidirectional else 1
        # ns = 2 if lstm_bidirectional else 1
        # re_input_size = self.tree_lstm_h_size * nt + 2 * seq_lstm_h_size * ns
        # tree_input_size = seq_lstm_h_size * ns + dep_embs_size + label_embs_size
        # self.tree_lstm = TreeLSTM(
        #                             embedding_dim=tree_input_size,
        #                             h_size=self.tree_lstm_h_size,
        #                             dropout=dropout,
        #                             bidirectional=tree_bidirectional
        #                           )
        # self.rel_decoder = nn.Sequential(
        #                                 nn.Linear(re_input_size, re_hidden_size), 
        #                                 nn.Tanh(), 
        #                                 self.dropout,
        #                                 nn.Linear(re_hidden_size, re_output_size)
        #                                 )




    def __schedule_sampling(self):
        schedule_sampling = self.k / (self.k +
                                    exp(batch.current_epoch / self.k))
        # schdule sampling
        coin_flip = floor(random() * 10) / 10
        
        return schdule_sampling > coin_flip:


    def __get_all_possible_pairs(span_lengths, none_unit_mask):
        
        batch_size = span_lengths.shape[0]
        end_idxs = torch.cumsum(span_lengths, dim=-1)
        
        all_possible_pairs = []
        for i in range(batch_size):
            idxes = end_idxs[i][none_unit_mask[i]]
            possible_pairs = list(itertools.product(idxes, repeat=2))
            
        return all_possible_pairs


    def forward(self, batch, output):

        #1)
        # pos_word_embs.shape = (batch_size, max_nr_tokens, word_embs + pos_embs)
        pos_word_embs = th.cat((batch["token"]["word_embs"], batch["token"]["pos_embs"]), dim=2)
        pos_word_embs = self.dropout(pos_word_embs)

        #2) lstm_out = (batch_size, max_nr_tokens, lstm_hidden)
        lstm_out, _ = self.lstm(pos_word_embs, batch["token"]["lengths"])

        #3)
        # logits = (batch_size, max_nr_tokens, nr_labels)
        # probs = (batch_size, max_nr_tokens, nr_labels)
        # preds = (batch_size, max_nr_tokens)
        # one_hots = (batch_size, max_nr_tokens, nr_layers)
        logits, probs, preds, one_hots = self.seg_label_clf(lstm_out, batch["token"]["lengths"])


        #4)
        if self.__schedule_sampling():
            preds = batch["token"]["seg+label"]
            one_hots = torch.one_hots(preds)


        #5)
        span_lengths, none_span_mask, nr_units = bio_decode( 
                                                            batch_encoded_bios=preds, 
                                                            lengths=batch["token"]["lengths"], 
                                                            apply_correction=True,
                                                            B=[], #ids for labels counted as B
                                                            I=[], #ids for labels counted as I
                                                            O=[], #ids for labels counted as O
                                                            )

        #6)
        #NOTE! we can change this to output a tensor or array if its suits better.
        all_possible_pairs = self.__get_all_possible_pairs(span_lengths, none_unit_mask)


        #7)
        node_embs = th.cat((lstm_out, one_hots, batch["token"]["dep_embs"]), dim=-1)



        #8) Im unsure how we are going to build the 
        # 

        graphs = self.build_dep_graphs(
                                        deplinks = batch["depheads"], 
                                        token_reps = node_embs, 
                                        subgraphs = all_possible_pairs
                                        )

        #9) 
        #
        tree_lstm_out = tree_lstm(graphs)

        #10) Here we should format the data to the following structure:
        # t1 = representation of the last token in the first unit of the pair
        # t2 = representation of the last token in the second unit of the pair
        # a = lowest ancestor of t1 and t2
        # (batch_size, nr_units, nr_units, a+t1+t2)
        pairs = ""

        #now we should get probabilites  for each link_labels
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
        link_label_score = self.link_label_clf(pairs)

        # 11)
        # first we get the index of the unit each unit links to
        # we do this by first get the highest score of the link label
        # for each unit pair. Then we argmax that to get the index of 
        # the linked unit.
        max_link_label_score = torch.max(link_label_score, dim=-1)
        link_preds = torch.argmax(max_link_label_score, dim=-1)

        # 12)
        # we index the link_label_scores by the link predictions, selecting
        # the logits for the link_labels for the linked pairs
        top_link_label_scores = index_4D(link_label_score, index=link_preds)

        #13) we mask out 
        unit_mask = unit_mask.type(torch.bool)
        pair_scores[~unit_mask]  =  float("inf")


        #



class S:

    def __init__(self):
        pass


    def forward(self, ):

