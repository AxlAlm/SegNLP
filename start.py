
# import segnlp
# from segnlp import Pipeline
# from segnlp.datasets.am import PE
# from segnlp.nn.models.general import LSTM_CRF
# from segnlp.features import GloveEmbeddings
# from segnlp.nn.default_hyperparamaters import get_default_hps


# from segnlp.utils import random_ints, set_random_seed
# import numpy as np

lstm_er = {
    "seq_lstm_h_size": 100,  # Sequential LSTM hidden size
    "tree_lstm_h_size": 100,  # Tree LSTM hidden size
    "ac_seg_hidden_size": 100,  # Entity recognition layer hidden size
    "re_hidden_size": 100,  # Relation extraction layer hidden size
    "seq_lstm_num_layers": 1,  # Sequential LSTM number of layer
    "lstm_bidirectional": True,  # Sequential LSTM bidirection
    "tree_bidirectional": True,  # Tree LSTM bidirection
    "k": 25,  # hyperparameter for scheduled sampling
    "graph_buid_type": 0,
    "sub_graph_type": 0,
    "dropout": 0.5,
    "optimizer": "adam",
    "lr": 0.001,
    "max_epochs": 300,
    "batch_size": 32,
    "gpus": 1
}


hps = [lstm_er, lstm_er]

dd = '\u2016'
keys = list(hps[0].keys()) + ["prog"]
row = "".join([f"|{{:7.5}}" for k in keys[:-1]]) + f"{dd}{{:8}}{dd}"

check = u'\u2713'
block = bytes((219,)).decode('cp437')
header = row.format(*keys)
print(header)
print(u"\u2017"*len(header))
for hp in hps:
    str_values = list(map(str,hp.values()))
    print(row.format(*str_values + [f"0/4 {check}"]))
    print(u"\u005F"*len(header))

# exp = Pipeline(
#                 project="debugging",
#                 dataset=PE( 
#                             tasks=["seg+label"],
#                             prediction_level="token",
#                             sample_level="document",
#                             ),
#                 features =[
#                             GloveEmbeddings(),
#                             ],
#                 model = LSTM_CRF
#             )


# print(exp.dataset.info)

# segnlp.settings["dl_n_workers"] = 1
# hps = get_default_hps(LSTM_CRF.name())
# hps["max_epochs"] = 1
# hps["lr"] = [0.01,0.001]

# exp.hp_tune(
#                 hyperparamaters = hps,
#                 ptl_trn_args=dict(
#                                     overfit_batches=0.1,
#                                     #gpus=[0]
#                                  )
#                 )