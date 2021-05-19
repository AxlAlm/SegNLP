
import sys
sys.path.insert(1, '../../')

from segnlp import Pipeline
from segnlp.datasets.am import MTC
from segnlp.nn.models.am import JointPN
from segnlp.features import GloveEmbeddings
from segnlp.features import UnitPos, BOW
from segnlp.nn.default_hyperparamaters import get_default_hps


exp = Pipeline(
                project="joint_np",
                dataset= MTC( 
                            tasks=["label", "link"],
                            prediction_level="unit",
                            sample_level="document",
                            ),
                features =[
                            GloveEmbeddings(),
                            UnitPos(),
                            BOW()
                            ],
                model = JointPN,
                evaluation_method="cross_validation"
            )

hps = {
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


best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            gpus=[1]
                                        ),
                        monitor_metric="val_link-f1"
                        )

# exp1_scores, exp1_outputs = exp.test()
# exp2_scores, exp2_outputs = exp.test(seg_preds="/tmp/seg_preds.csv")
