
from hotam import ExperimentManager
from hotam.datasets import PE
from hotam.nn.models import LSTM_CRF
from hotam.features import Embeddings
from hotam.dashboard import DummyDash
from hotam.database import MongoDB
from hotam.loggers import MongoLogger

if __name__ == "__main__":

	db = MongoDB()
	dashboard = DummyDash(db=db).run_server(
											port=8050,
											debug=True,
											use_reloader=False,
											)
	exp_logger = MongoLogger(db=db)


	pe = PE()
	pe.setup(
			tasks=["seg"],
			multitasks=[], 
			sample_level="sentence",
			prediction_level="token",
			encodings=[],
			features=[
						Embeddings("glove")
						],
			remove_duplicates=False,
			)
	
	hp =  {
			"optimizer": "sgd",
			"lr": 0.001,
			"hidden_dim": 256,
			"num_layers": 2,
			"bidir": True,
			"fine_tune_embs": False,
			"batch_size": 10,
			"max_epochs":10,
			}

	ta = {
			"logger":None,
			"checkpoint_callback":False,
			"early_stop_callback":False,
			"progress_bar_refresh_rate":1,
			"check_val_every_n_epoch":1,
			"gpus":None,
			#"gpus": [1],
			"num_sanity_val_steps":1,  
			"overfit_batches":0.01
			}


	M = ExperimentManager()
	M.run( 
			project="seg",
			dataset=pe,
			model=LSTM_CRF,
			hyperparamaters=hp,
			trainer_args=ta,
			monitor_metric="val-seg-f1",
			progress_bar_metrics=["val-seg-f1"],
			exp_logger=exp_logger,
			debug_mode=False,
			)
