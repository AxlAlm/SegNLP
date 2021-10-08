
#basics
import warnings
import os
import multiprocessing

#segnlp
from .logger import get_logger
from .pipeline import Pipeline
from .utils import set_random_seed

#sklearn
from sklearn.exceptions import UndefinedMetricWarning

# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore")

settings = {
            "dl_n_workers": 0 #multiprocessing.cpu_count()
            }

set_random_seed(42)

__version__ = 0.1
# __all__ = [
#             "get_logger",
#             "Pipeline",
#             ]