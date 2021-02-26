# from hotam.default_hyperparamaters.lstm_crf import lstm_crf_hps
# from hotam.default_hyperparamaters.lstm_crf import lstm_cnn_crf_hps
# from hotam.default_hyperparamaters.lstm_crf import joint_pointer_nn_hps
# from hotam.default_hyperparamaters.lstm_crf import lstm_dist_hps

lstm_crf_hps = {
    "optimizer": "sgd",
    "lr": 0.001,
    "hidden_dim": 256,
    "num_layers": 2,
    "bidir": True,
    "fine_tune_embs": False,
    "batch_size": 32,
    "max_epochs": 100,
}

lstm_cnn_crf_hps = {
    "optimizer": "adam",
    "lr": 0.001,
    "hidden_dim": 250,
    "char_dim": 100,
    "kernel_size": 3,
    "num_layers": 1,
    "bidir": True,
    "batch_size": 32,
    "max_epochs": 100,
}

joint_pointer_nn_hps = {
    "optimizer": "adam",
    "lr": 0.001,
    "encoder_input_dim": 256,
    "encoder_hidden_dim": 256,
    "encoder_num_layers": 2,
    "encoder_bidir": True,
    "decoder_hidden_dim": 512,
    "feature_dropout": 0.8,
    "encoder_dropout": 0.8,
    "decoder_dropout": 0.8,
    "task_weight": 0.5,
    "batch_size": 16,
    "max_epochs": 1000,
}

lstm_dist_hps = {
    "optimizer": "sgd",
    "lr": 0.001,
    "hidden_dim": 256,
    "num_layers": 1,
    "bidir": True,
    "batch_size": 10,
    "max_epochs": 10,
    "alpha": 0.5,
    "beta": 0.5
}

lstm_er = {
    "dep_embedding_size": 50,  # Embed dimension for depedency label
    "seq_lstm_h_size": 128,  # Sequential LSTM hidden size
    "tree_lstm_h_size": 128,  # Tree LSTM hidden size
    "ner_hidden_size": 45,  # Entity recognition layer hidden size
    "ner_output_size": 30,  # Entity recognition layer output size
    "re_hidden_size": 30,  # Relation extraction layer hidden size
    "re_output_size": 30,  # Relation extraction layer output size
    "seq_lstm_num_layers": 1,  # Sequential LSTM number of layer
    "lstm_bidirectional": True,  # Sequential LSTM bidirection
    "tree_bidirectional": True,  # Tree LSTM bidirection
    "k": 10,  # hyperparameter for scheduled sampling
    "graph_buid_type": 0,
    "sub_graph_type": 0,
    "dropout": 0.5,
    "optimizer": "adam",
    "lr": 0.0001,
    "max_epochs": 10,
    "batch_size": 10,
}

dummy_hps = {
    "optimizer": "adam",
    "lr": 0.001,
    "hidden_dim": 100,
    "num_layers": 1,
    "batch_size": 32,
    "max_epochs": 100,
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
