
from hotam import ExperimentManager
from hotam.datasets import PE
from hotam.nn.models import LSTM_CRF
from hotam.nn.models import LSTM_CNN_CRF
from hotam.nn.models import JointPN
from hotam.nn.models import LSTM_DIST


from hotam.features import Embeddings, DocPos
from hotam.dashboard import FullDash
from hotam.database import MongoDB
from hotam.loggers import MongoLogger

if __name__ == "__main__":

	db = MongoDB()
	dashboard = FullDash(db=db).run_server(
											port=8050,
											debug=True,
											#use_reloader=False,
											)
	#exp_logger = MongoLogger(db=db)

	# pe = PE()
	# pe.setup(
	# 		tasks=["seg"],
	# 		multitasks=[], 
	# 		sample_level="sentence",
	# 		prediction_level="token",	
	# 		encodings=[],
	# 		features=[
	# 					Embeddings("glove"),
	# 					Embeddings("flair"),
	# 					Embeddings("bert"),
	# 					],
	# 		#remove_duplicates=False,
	# 		#tokens_per_sample=True,
	# 		#override=True
	# 		)


	# M = ExperimentManager()
	# M.run( 
	# 		project="seg",
	# 		dataset=pe,
	# 		model=LSTM_CRF,
	# 		monitor_metric="val-seg-f1",
	# 		progress_bar_metrics=["val-seg-f1"],
	# 		exp_logger=exp_logger,
	# 		debug_mode=False,
	# 		)
