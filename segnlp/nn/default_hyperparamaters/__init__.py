# from segnlp.default_hyperparamaters.lstm_crf import lstm_crf_hps
# from segnlp.default_hyperparamaters.lstm_crf import lstm_cnn_crf_hps
# from segnlp.default_hyperparamaters.lstm_crf import joint_pointer_nn_hps
# from segnlp.default_hyperparamaters.lstm_crf import lstm_dist_hps

lstm_crf_hps = {
    "optimizer": "sgd",
    "lr": 0.1,
    "hidden_dim": 256,
    "num_layers": 2,
    "bidir": True,
    "fine_tune_embs": True,
    "batch_size": 32,
    "max_epochs": 150,
    "scheduler": "rop",
    "patience": 10
}

lstm_cnn_crf_hps = {
    "optimizer": "sgd",
    "lr": 0.1,
    "hidden_dim": 200, #[125, 150, 200,250],
    "char_dim": 30,
    "n_filters": 20,
    "kernel_size": 3,
    "num_layers": 1,
    "bidir": True,
    "batch_size": 32,
    "max_epochs": 200,
    "dropout": 0.5,
    "patience": 5,
    
}

joint_pointer_nn_hps = {
    "optimizer": "adam",
    "lr": 0.001,
    "encoder_hidden_dim": 256,
    "encoder_num_layers": 1,
    "encoder_bidir": True,
    "decoder_hidden_dim": 512,
    "encoder_dropout": 0.9,
    "decoder_dropout": 0.9,
    "task_weight": 0.5,
    "batch_size": 16,
    "max_epochs": 4000,
    "patience": 15,

}

lstm_dist_hps = {
    "optimizer": "adam",
    "lr": 0.001,
    "hidden_dim": 256,
    "num_layers": 1,
    "bidir": True,
    "batch_size": 16,
    "max_epochs": 500,
    "alpha": 0.5,
    "beta": 0.25,
    "input_dropout": 0.1,
    "lstm_dropout" : 0.1,
    "output_dropout": 0.5,
    #"patience": 10

}

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
}

dummy_hps = {
    "optimizer": "adam",
    "lr": 0.001,
    "hidden_dim": 100,
    "num_layers": 1,
    "batch_size": 32,
    "max_epochs": 10,
}


def get_default_hps(model_name):

    if model_name.lower() == "lstm_crf":
        return lstm_crf_hps
    elif model_name.lower() == "lstm_cnn_crf":
        return lstm_cnn_crf_hps
    elif model_name.lower() == "jointpn":
        return joint_pointer_nn_hps
    elif model_name.lower() == "lstm_dist":
        return lstm_dist_hps
    elif model_name.lower() == "dummynn":
        return dummy_hps
    elif model_name.lower() == "lstm_er":
        return lstm_er
    else:
        raise NotImplementedError
