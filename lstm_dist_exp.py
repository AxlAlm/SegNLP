
import segnlp
from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.nn.models.am import LSTM_DIST
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

segnlp.settings["dl_n_workers"] = 0
hps = get_default_hps(LSTM_DIST.name())
best_hp = exp.hp_tune(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            gpus=[0]
                                        )
                        )

