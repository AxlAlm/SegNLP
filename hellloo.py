
from hotam import ExperimentManager
from hotam.datasets import PE
from hotam.nn.models import LSTM_CRF
from hotam.nn.models import LSTM_CNN_CRF
from hotam.nn.models import JointPN


from hotam.features import Embeddings, DocPos
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
			tasks=["ac", "relation"],
			multitasks=[], 
			sample_level="paragraph",
			prediction_level="ac",
			encodings=[],
			features=[
						DocPos(pe, prediction_level="sentence"),
						Embeddings("glove")
						],
			remove_duplicates=False,
			)


	M = ExperimentManager()
	M.run( 
			project="ac-relation",
			dataset=pe,
			model=JointPN,
			monitor_metric="val-seg-f1",
			progress_bar_metrics=["val-seg-f1"],
			exp_logger=exp_logger,
			debug_mode=False,
			)
