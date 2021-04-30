

import unittest
import pandas as pd
from segnlp.datasets import PE, MTC
from segnlp.datasets import DataSet
import itertools


class TestDataset(unittest.TestCase):

    def instance(self, dataset):
        self.assertIsInstance(dataset, DataSet)

    def attributes(self, dataset):
        self.assertIsInstance(dataset.tasks, list)
        self.assertIsInstance(dataset.task_labels, dict)
        self.assertIsInstance(dataset.splits, dict)


    def data(self, dataset):
        self.assertIsInstance(dataset.data, pd.DataFrame)
        self.assertIsNot(dataset.data.shape,(0,0))
        self.assertEqual(list(dataset.data.columns), ["sample_id", "text", "text_type", "span_labels"])


class TestPE(TestDataset):

    def test_ok_kwargs(self):
        should_work = [
                        {
                            "tasks":["seg", "label", "link_label", "link"],
                            "sample_level": "document",
                            "prediction_level":"token",
                        },
                        {
                            "tasks":["seg+label+link_label+link"],
                            "sample_level": "paragraph",
                            "prediction_level":"token",
                        },
                        {
                            "tasks":["label","link+link_label"],
                            "sample_level": "document",
                            "prediction_level":"unit",
                        },
                        {
                            "tasks":["seg+label"],
                            "sample_level": "sentence",
                            "prediction_level":"token",
                        }  
                        ]


        for kwargs in should_work:
            dataset = PE(**kwargs)
            self.instance(dataset)
            self.attributes(dataset)
            self.data(dataset)


    def test_bad_kwargs(self):
        should_not_work = [
                            (
                            {
                            "tasks":["seg", "seg+link"],
                            "sample_level": "document",
                            "prediction_level":"token",
                            },"testing duplicate tasks"
                            ),
                            (
                            {
                            "tasks":["label", "label", "seg"],
                            "sample_level": "paragraph",
                            "prediction_level":"token",
                            },"testing duplicate tasks"
                            ),
                            (
                            {
                            "tasks":["label", "link_label", "link"],
                            "sample_level": "document",
                            "prediction_level":"token",
                            }, "testing missing segmentation task"
                            ),
                            (
                            {
                            "tasks":["seg"],
                            "sample_level": "ddfdocument",
                            "prediction_level":"unit",
                            }, "testing unsupported sample level"
                            ),
                            (
                            {
                            "tasks":["label", "link_ladfdbel", "lisssnk"],
                            "sample_level": "document",
                            "prediction_level":"unit",
                            }, "testing unsupported task"
                            ),
                            (
                            {
                            "tasks":["label"],
                            "sample_level": "document",
                            "prediction_level":"paragraph",
                            }, "testing unsupported prediction level"
                            )
                            ]

        for kwargs, msg in should_not_work:
            with self.assertRaises(RuntimeError, msg=msg) as e:
                PE(**kwargs)



class TestMTC(TestDataset):

    def test_ok_kwargs(self):
        should_work = [
                        {
                            "tasks":["seg", "label", "link_label", "link"],
                            "sample_level": "paragraph",
                            "prediction_level":"token",
                        },
                        {
                            "tasks":["seg+label+link_label+link"],
                            "sample_level": "sentence",
                            "prediction_level":"token",
                        },
                        {
                            "tasks":["label","link+link_label"],
                            "sample_level": "paragraph",
                            "prediction_level":"unit",
                        }      
                        ]


        for kwargs in should_work:
            dataset = MTC(**kwargs)
            self.instance(dataset)
            self.attributes(dataset)
            self.data(dataset)


    def test_bad_kwargs(self):
        should_not_work = [
                            (
                            {
                            "tasks":["seg", "seg+link"],
                            "sample_level": "paragraph",
                            "prediction_level":"token",
                            },"testing duplicate tasks"
                            ),
                            (
                            {
                            "tasks":["label", "label", "seg"],
                            "sample_level": "paragraph",
                            "prediction_level":"token",
                            },"testing duplicate tasks"
                            ),
                            (
                            {
                            "tasks":["label", "link_label", "link"],
                            "sample_level": "paragraph",
                            "prediction_level":"token",
                            }, "testing missing segmentation task"
                            ),
                            (
                            {
                            "tasks":["seg"],
                            "sample_level": "document",
                            "prediction_level":"unit",
                            }, "testing unsupported sample level"
                            ),
                            (
                            {
                            "tasks":["label", "link_ladfdbel", "lisssnk"],
                            "sample_level": "paragraph",
                            "prediction_level":"unit",
                            }, "testing unsupported task"
                            ),
                            (
                            {
                            "tasks":["label"],
                            "sample_level": "paragraph",
                            "prediction_level":"paragraph",
                            }, "testing unsupported prediction level"
                            )
                            ]

        for kwargs, msg in should_not_work:
            with self.assertRaises(RuntimeError, msg=msg) as e:
                MTC(**kwargs)



if __name__=='__main__':
    unittest.main()