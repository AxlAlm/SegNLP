
# from hotam.default_hyperparamaters.lstm_crf import lstm_crf_hps
# from hotam.default_hyperparamaters.lstm_crf import lstm_cnn_crf_hps
# from hotam.default_hyperparamaters.lstm_crf import joint_pointer_nn_hps
# from hotam.default_hyperparamaters.lstm_crf import lstm_dist_hps



lstm_crf_hps =  {
                    "optimizer": "sgd",
                    "lr": 0.001,
                    "hidden_dim": 256,
                    "num_layers": 2,
                    "bidir": True,
                    "fine_tune_embs": False,
                    "batch_size": 10,
                    "max_epochs":100,
                    }
            

lstm_cnn_crf_hps = {
                    "optimizer": "sgd",
                    "lr": 0.001,
                    "hidden_dim": 256,
                    "char_dim": 100,
                    "kernel_size": 3,
                    "num_layers": 2,
                    "bidir": True,
                    "batch_size": 10,
                    "max_epochs":10,
                    }


joint_pointer_nn_hps = {
                        "optimizer": "sgd",
                        "lr": 0.001,
                        "encoder_input_dim": 256,
                        "encoder_hidden_dim": 256,
                        "encoder_num_layers":2,
                        "encoder_bidir":True,
                        "decoder_hidden_dim": 512,
                        "feature_dropout": 0.8,
                        "encoder_dropout": 0.8,
                        "decoder_dropout": 0.8,
                        "task_weight":0.5,
                        "batch_size": 20,
                        "max_epochs":10,
                        }

lstm_dist_hps = {
                "optimizer": "sgd",
                "lr": 0.001,
                "hidden_dim": 256,
                "num_layers": 1,
                "bidir": True,
                "batch_size": 10,
                "max_epochs":10,
                "alpha": 0.5,
                "beta": 0.5
                }


def get_default_hps(model_name):
    
    if model_name.lower() == "lstm_crf":
        return lstm_crf_hps
    elif model_name.lower() == "lstm_cnn_crf":
        return lstm_cnn_crf_hps
    elif model_name.lower() == "jointpn":
        return joint_pointer_nn_hps
    elif model_name.lower()== "lstm_dist":
        return lstm_dist_hps
    else:
        raise NotImplementedError