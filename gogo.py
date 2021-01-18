
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

from tqdm import tqdm


if __name__ == "__main__":

	db = MongoDB()
	exp_logger = MongoLogger(db=db)

	pe = PE()
	#pe.example(sample_id=928, level="paragraph")
	#pe.example(sample_id=928, level="paragraph")

	pe.setup(
			tasks=["ac", "relation"],
			multitasks=[], 
			sample_level="paragraph",
			prediction_level="ac",	
			encodings=["pos", "deprel", "dephead"],
			features=[
						#DocPos(dataset=pe, prediction_level="sentence")
						],
			#remove_duplicates=False,
			#tokens_per_sample=True,
			#override=True
			)
	
	d = pe[[1,2]]
	print(pe.decode_list(d["deprel"][0][1], "deprel"))
	print(d["sent2root"][0])

	
	# pe.example(sample_id=928, level="paragraph")

	# M = ExperimentManager()
	# M.run( 
	# 		project="pe_seg_ac",
	# 		dataset=pe,
	# 		model=LSTM_CNN_CRF,
	# 		monitor_metric="val-seg_ac-f1",
	# 		progress_bar_metrics=["val-seg_ac-f1"],
	# 		exp_logger=exp_logger,
	# 		debug_mode=False,
	# 		)
