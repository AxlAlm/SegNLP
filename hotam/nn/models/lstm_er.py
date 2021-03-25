

from hotam.nn.layer.seg_layers import BigramSegLayer
from hotam.nn.layers.link_label_layers import DepPairingLayer
from hotam.nn.layers.lstm import LSTM_LAYER
from hotam.nn.utils import index_4D, get_all_possible_pairs

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
    
        self.link_label_clf = self.DepPairingLayer()



    def forward(self, batch, output):

        #1)
        # pos_word_embs.shape = (batch_size, max_nr_tokens, word_embs + pos_embs)
        pos_word_embs = th.cat((batch["token"]["word_embs"], batch["token"]["pos_embs"]), dim=2)
        pos_word_embs = self.dropout(pos_word_embs)

        #2) lstm_out = (batch_size, max_nr_tokens, lstm_hidden)
        lstm_out, _ = self.lstm(pos_word_embs, batch["token"]["lengths"])

        #3)
        # seg_label_logits = (batch_size, max_nr_tokens, nr_labels)
        # seg_label_probs = (batch_size, max_nr_tokens, nr_labels)
        # seg_label_preds = (batch_size, max_nr_tokens)
        # one_hots = (batch_size, max_nr_tokens, nr_layers)
        seg_label_logits, seg_label_probs, seg_label_preds, one_hots = self.seg_label_clf(lstm_out, batch["token"]["lengths"])


        #4)
        if schedule_sampling(batch.current_epoch, k):
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
        all_possible_pairs = get_all_possible_pairs(span_lengths, none_unit_mask)

        #7)
        node_embs = th.cat((lstm_out, one_hots, batch["token"]["dep_embs"]), dim=-1)


        #8)
        link_label_logits, link_preds = self.link_label_clf(
                                                            input_embs = node_embs,
                                                            dependecies = batch["token"]["dephead"],
                                                            pairs = all_possible_pairs,
                                                            mode = "shortest_path"
                                                            )       

        if self.train_mode:

            #CALCULATE LOSS HERE


            output.add_loss(task="total",       data=total_loss)
            output.add_loss(task="link_label",  data=link_label_loss)
            output.add_loss(task="label",       data=label_loss)


        output.add_preds(task="seg+label",      level="token", data=label_preds)
        output.add_preds(task="link",           level="unit", data=link_preds)
        output.add_preds(task="link_label",     level="unit", data=link_label_preds)

        return output

       