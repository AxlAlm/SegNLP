

import unittest
import pandas as pd
from hotam.datasets import PE, MTC


class TestDataset(unittest.TestCase):


    def test_datasets(self):
        for dataset_class in [PE, MTC]:
            dataset = dataset_class() 
            self.attributes(dataset)
            self.data(dataset)
    

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