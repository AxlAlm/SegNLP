

import unittest
import pandas as pd
from hotam.preprocessing import Preprocessor


class TestDataset(unittest.TestCase):

        #prediction_level="",
        #sample_level:str,
        #input_level:str,

    def test_init(self):
        Preprocessor(
                    features:list = [],
                    encodings:list = [],
                    tokens_per_sample:bool=False,
                    argumentative_markers:bool=False
                    )

    def attributes(self, dataset):
        self.assertIsInstance(dataset.tasks, list)
        self.assertIsInstance(dataset.task_labels, dict)
        self.assertIsInstance(dataset.splits, dict)


    def data(self, dataset):
        self.assertIsInstance(dataset.data, pd.DataFrame)
        self.assertIsNot(dataset.data.shape,(0,0))
        self.assertEqual(list(dataset.data.columns), ["sample_id", "text", "text_type", "span_labels"])


if __name__=='__main__':
    unittest.main()