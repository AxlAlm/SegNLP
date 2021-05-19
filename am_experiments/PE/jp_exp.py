
import sys
sys.path.insert(1, '../../')

from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.nn.models.am import JointPN
from segnlp.features import GloveEmbeddings
from segnlp.features import UnitPos, BOW
from segnlp.nn.default_hyperparamaters import get_default_hps


exp = Pipeline(
                project="joint_np",
                dataset=PE( 
                            tasks=["label", "link"],
                            prediction_level="unit",
                            sample_level="paragraph",
                            ),
                features =[
                            GloveEmbeddings(),
                            UnitPos(),
                            BOW()
                            ],
                model = JointPN
            )

hps = get_default_hps(JointPN.name())
best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            gpus=[2]
                                        ),
                        monitor_metric="val_link-f1"
                        )

exp1_scores, exp1_outputs = exp.test()
exp2_scores, exp2_outputs = exp.test(seg_preds="/tmp/seg_preds.csv")
