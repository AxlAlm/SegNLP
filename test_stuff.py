

# lstm_er = {
#             "seq_lstm_h_size": 100,  # Sequential LSTM hidden size
#             "tree_lstm_h_size": 100,  # Tree LSTM hidden size
#             "ac_seg_hidden_size": 100,  # Entity recognition layer hidden size
#             "re_hidden_size": 100,  # Relation extraction layer hidden size
#             "seq_lstm_num_layers": 1,  # Sequential LSTM number of layer
#             "lstm_bidirectional": True,  # Sequential LSTM bidirection
#             "tree_bidirectional": True,  # Tree LSTM bidirection
#             "k": 25,  # hyperparameter for scheduled sampling
#             "graph_buid_type": 0,
#             "sub_graph_type": 0,
#             "dropout": 0.5,
#             "optimizer": "adam",
#             "lr": 0.001,
#             "max_epochs": 300,
#             "batch_size": 32,
#             "gpus": 1
#         }

# hps = [lstm_er] * 10


def hp_tune_tabel(
                    ongoing: dict,
                    top: dict,
                    to_do:list,
                    n_seeds=4,
                    show_n=3
                    ):
    
    double_line = '\u2016'
    keys = ["i"] + list(ongoing.keys()) + ["prog"]
    row = "{} "+"".join([f"|{{:^6.5}}" for k in keys[1:-1]]) + double_line + f"{double_line}{{:^8}}"*2 + double_line
    check = u'\u2713'
    block = bytes((219,)).decode('cp437')

    header = row.format(*keys)
    print("----------Hyperparamaters-----------")
    print(header)
    size = len(header)
    print(u"\u2017"*size)
    more = False
    
    str_values = list(map(str,ongoing.values()))
    row_values =  [str(ongoing["id"])] + str_values + [f"{ongoing["progress"]}/4", f"{ongoing["rank"]}"]
    r = row.format(*row_values)
    print(r)
    print(u"\u2017"*size)

    str_values = list(map(str,top.values()))
    row_values =  [str(top["id"])] + str_values + [f"{top["progress"]}/4", f"{top["rank"]}"]
    r = row.format(*row_values)
    print(r)
    print(u"\u2017"*size)

    for hp in to_do:

        if hp["progress"] == n_seeds:
            p = check
        else:
            p = f"{hp["progress"]}/4"

        str_values = list(map(str,hp.values()))
        row_values =  [str(hp["id"])] + str_values + [f"{p}"]
        r = row.format(*row_values)
        print(r)

        if i >= show_n:
            more = True
            break
        
        print(u"\u005F"*size)

    if more:
        print("......")
        print(u"\u005F"*len(header))
