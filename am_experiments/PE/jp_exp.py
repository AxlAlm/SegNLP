
import sys
sys.path.insert(1, '../../')

from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.models import JointPN
from segnlp.pretrained_features import GloveEmbeddings
from segnlp.pretrained_features import SegPos
from segnlp.resources.vocab import bnc_vocab



exp = Pipeline(
                id="jp_pe",
                dataset=PE( 
                            tasks=["label", "link"],
                            prediction_level="seg",
                            sample_level="paragraph",
                            ),
                model = JointPN,
                metric = "default_segment_metric",
                pretrained_features =[
                                        GloveEmbeddings(),
                                        ],
                #overwrite = True
            )

hps = {
        "general":{
                "optimizer": "Adam",
                "lr": 0.001,
                "batch_size": 16,
                "max_epochs":4000,
                "patience": 15,
                "task_weight": 0.5,
                },
        "Agg":{
                "mode":"mix",
                },
        "SegBOW":{
                "vocab": bnc_vocab(size = 10000),
                },
        "LinearRP": {
                "hidden_size": 512,
                #"activation": "Sigmoid"

                },
        "encoder_LSTM": {  
                    "dropout":0.9,
                    "hidden_size": 256,
                    "num_layers":1,
                    "bidir":True,
                    },
        "decoder_LSTM": {  
                    "dropout":0.9,
                    "hidden_size": 512,
                    "num_layers":1,
                    "bidir":False,

                    },
        "Pointer": {
                    }
        }

best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            #gpus=[1]
                                        ),
                        monitor_metric="val_link-f1"
                        )

exp1_scores, exp1_outputs = exp.test()
# exp2_scores, exp2_outputs = exp.test(seg_preds="/tmp/seg_preds.csv")
