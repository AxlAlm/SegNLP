
from hotam.nn.models.lstm_crf import LSTM_CRF
from hotam.nn.models.lstm_cnn_crf import LSTM_CNN_CRF
from hotam.nn.models.joint_pointer_nn import JointPN
from hotam.nn.models.lstm_dist import LSTM_DIST
from hotam.nn.models.dummy_nn import DummyNN


__all__ = [ 
            
            "DummyNN",
            "LSTM_DIST",
            "LSTM_CRF",
            "LSTM_CNN_CRF",
            "JointPN",
            ]  


def get_model(model_name):
    for model in __all__:
        if model.name() == model_name:
            return model

    raise KeyError(f"No model named {model_name}")
