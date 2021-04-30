
sys.path.insert(1, '../')


from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.nn.models.am import LSTM_DIST
from segnlp.features import GloveEmbeddings
from segnlp.features import UnitPos, BOW
from segnlp.nn.default_hyperparamaters import get_default_hps


exp = Pipeline(
                project="LSTM_DIST",
                dataset=PE( 
                            tasks=["label", "link", "link_label"],
                            prediction_level="unit",
                            sample_level="paragraph",
                            ),
                features =[
                            GloveEmbeddings(),
                            UnitPos(),
                            BOW()
                            ],
                model = LSTM_DIST,
                other_levels = ["am"]
            )

hps = get_default_hps(LSTM_DIST.name())
best_hp = exp.hp_tune(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            gpus=[0]
                                        )
                        )

