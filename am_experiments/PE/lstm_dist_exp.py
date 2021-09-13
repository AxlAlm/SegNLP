

import sys
sys.path.insert(1, '../../')


from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.pretrained_features import ELMoEmbeddings
from segnlp.resources.vocab import bnc_vocab


exp = Pipeline(
                id="lstm_dist_pe",
                dataset=PE( 
                            tasks=["label", "link", "link_label"],
                            prediction_level="seg",
                            sample_level="paragraph",
                            ),
                pretrained_features = [
                                ELMoEmbeddings(),
                                ],
                model = "LSTM_DIST",
                other_levels = ["am"],
                metric = "default_segment_metric",
                #overwrite = True
            )


lstm_hps = {   
        "input_dropout": 0.1,
        "dropout":0.1,
        "hidden_size": 256,
        "num_layers":1,
        "bidir":True,
        "weight_init": "orthogonal_",
        }

linear_clf_hps = {
                "weight_init": "uniform_",
                "weight_init_kwargs": {"a": -0.05, "b": 0.05},
                }

hps = {
        "general":{
                "optimizer": "Adam",
                "lr": 0.001,
                "batch_size": 16,
                "max_epochs":500,
                "patience": 10,
                "task_weight": 0.5,
                },
        "SegBOW":{
                "vocab": bnc_vocab(size = 10000),
                "mode": "counts",
                },
        "BOW_DimRedu":{
                        "out_dim": 300,
                        "weight_init": "uniform_",
                        "weight_init_kwargs": {"a": -0.05, "b": 0.05}
                        },
        "Word_LSTM": lstm_hps,
        "AM_LSTM": lstm_hps,
        "AC_LSTM": lstm_hps,
        "ADU_LSTM": {**{"output_dropout": 0.5}, **lstm_hps},
        "Link_LSTM": {**{"output_dropout": 0.5}, **lstm_hps},
        "Link_label_LinearCLF": linear_clf_hps,
        "Label_LinearCLF": linear_clf_hps,
        "Pairer": {
                    "mode": ["cat", "multi"],
                    "n_rel_pos": 25,
                    },
                
        }


best_hp = exp.train(
                        hyperparamaters = hps,
                        n_random_seeds=6,
                        ptl_trn_args=dict(
                                            gpus=[1,2],
                                            accelerator='ddp',
                                            replace_sampler_ddp = False
                                           #overfit_batches = 0.1

                                        ),
                        monitor_metric="val_f1",

                        )

#exp1_scores, exp1_outputs = exp.test()

#seg_pred = pd.read_csv("/tmp/pred_segs.csv")
# print(exp1_scores)
#exp2_scores, exp2_outputs = exp.test(seg_preds=seg_pred)
