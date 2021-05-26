
from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.models import JointPN
from segnlp.features import GloveEmbeddings
from segnlp.features import SegPos
from segnlp.features import BOW

exp = Pipeline(
                id="jp_pe",
                dataset=PE( 
                            tasks=["label", "link"],
                            prediction_level="seg",
                            sample_level="paragraph",
                            ),
                model = JointPN,
                metric = "default_segment_metric",
                features =[
                            GloveEmbeddings(),
                            SegPos(),
                            BOW()
                            ],
            )

hps = {
        "general":{
                "optimizer": "Adam",
                "lr": 0.001,
                "batch_size": 16,
                "max_epochs": 4000,
                "patience": 15,
                "task_weight": 0.5,
                },
        "Agg":{
                "mode":"mix",
                },
        "LLSTM": {  
                    "dropout":0.9,
                    "hidden_size": 256,
                    "num_layers":1,
                    "bidir":True,
                    },
        "Pointer": {
                    "dropout":0.9,
                    "hidden_size":512,
                    }
        }

best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            #gpus=[2]
                                        ),
                        monitor_metric="val_link-f1"
                        )

# exp1_scores, exp1_outputs = exp.test()
# exp2_scores, exp2_outputs = exp.test(seg_preds="/tmp/seg_preds.csv")
