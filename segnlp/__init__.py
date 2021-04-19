

from .logger import get_logger
#from segnlp.preprocessing import Preprocessor
from .pipeline import Pipeline, ChainedPipeline
from .utils import set_random_seed

set_random_seed(42)

__version__ = 0.1
__all__ = [
            "get_logger",
            "Pipeline",
            "ChainedPipeline"
            ]