
#basics
from typing import Sequence
import unittest
import os
import pandas as pd
import numpy as np

# segnlp
from segnlp.nlp import NLP
from segnlp.utils import RangeDict
from segnlp.utils import load_json


class TestSample(unittest.TestCase):


    def test_nlp_sample_creation(self):
        doc = "This is a not a segment.23 However, here starts a segement. This, right after this comma, is also a segment."
        nlp = NLP()
        self.sample = nlp(doc)


    def test_span_label(self):

        span2label = RangeDict({
            (0, 33): {'span_id': 0, 'seg_id': -1, 'label': 'None', 'link': -1, 'link_label': 'None'}, 
            (34, 57): {'span_id': 1, 'seg_id': 0, 'label': 'A', 'link': 1, 'link_label': 'A'},
            (58, 87): {'span_id': 2, 'seg_id': -1, 'label': 'None', 'link': -1, 'link_label': 'None'}, 
            (88, 106): {'span_id': 3, 'seg_id': 1, 'label': 'B', 'link': 1, 'link_label': 'root'}, 
        })
        task_labels = {
                        "seg": ["O", "B", "I"],
                        "label": ["None", "A", "B"],
                        "link_label": ["root", "A"]
                        }
        self.sample.add_span_labels(
                                    span2label, 
                                    task_labels = task_labels, 
                                    label_ams = True
                                    )


    def test_sample_df(self):
        path_to_gold_sample = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gold_sample.csv")
        self.gold_df = pd.read_csv(path_to_gold_sample, index_col = 0)
        is_equal = self.gold_df.equals(self.sample.df)
        self.assertTrue(is_equal)


    def test_sample_str(self):
        correct_str = "this is a not a segment . however , here starts a segement . this , right after this comma , is also a segment ."
        self.assertEqual(str(self.sample), correct_str)


    def test_token_length(self):
        length = self.sample.get("token", "length")
        self.assertIsInstance(length, int)
        self.assertEqual(length, 26)


    def test_seg_length(self):
        length = self.sample.get("seg", "length")
        self.assertIsInstance(length, int)
        self.assertEqual(length, 2)


    def test_seg_span_idxs(self):
        span_idxs = self.sample.get("seg","span_idxs")
        self.assertIsInstance(span_idxs, np.ndarray)
        self.assertEqual(span_idxs, np.array([[9, 13], [21, 25]]))


    def test_am_span_idxs(self):
        span_idxs = self.sample.get("am","span_idxs")
        self.assertIsInstance(span_idxs, np.ndarray)
        self.assertEqual(span_idxs, np.array([[7, 8], [14, 20]]))


    def test_token_df_values(self):

        for c in self.gold_df.columns:
            values = self.sample.get("token", c)
            self.assertIsInstance(values, np.ndarray)
            is_equal = np.array_equal(values, self.gold_df[c].to_numpy())
            self.assertTrue(is_equal)


