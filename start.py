
from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.models import JointPN
from segnlp.features import GloveEmbeddings
from segnlp.features import UnitPos
from segnlp.features import BOW

exp = Pipeline(
                project="jp_pe",
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

# best_hp = exp.train(
#                         hyperparamaters = hps,
#                         n_random_seeds=6,
#                         ptl_trn_args=dict(
#                                             gpus=[2]
#                                         ),
#                         monitor_metric="val_link-f1"
#                         )

# exp1_scores, exp1_outputs = exp.test()
# exp2_scores, exp2_outputs = exp.test(seg_preds="/tmp/seg_preds.csv")
