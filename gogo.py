
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
	exp_logger = MongoLogger(db=db)

	pe = PE()
	pe.setup(
			tasks=["seg_ac_relation_stance"],
			multitasks=[], 
			sample_level="document",
			prediction_level="token",	
			encodings=["chars"],
			features=[
						Embeddings("glove"),
						],
			#remove_duplicates=False,
			#tokens_per_sample=True,
			#override=True
			)

	M = ExperimentManager()
	M.run( 
			project="pe_end-to-end",
			dataset=pe,
			model=LSTM_CNN_CRF,
			monitor_metric="val-seg_ac_relation_stance-f1",
			progress_bar_metrics=["val-seg_ac_relation_stance-f1"],
			exp_logger=exp_logger,
			debug_mode=False,
			)
