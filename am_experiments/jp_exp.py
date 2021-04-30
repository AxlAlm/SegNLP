
import segnlp
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

exp.dataset.info
hps = get_default_hps(JointPN.name())
best_hp = exp.hp_tune(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            gpus=[0]
                                        )
                        )

