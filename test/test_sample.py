
#basics
import unittest
import os
import pandas as pd
import numpy as np

#pytroch
import torch
from torch._C import Value


# segnlp
from segnlp.data import Sample
from utils import Utils


# make sure unitest doesnt sort our tests
#unittest.TestLoader.sortTestMethodsUsing = None


class TestSample(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.utils = Utils() 


    def test_sample_creation(self):
        sample = self.utils.create_sample()
        self.assertIsInstance(sample , Sample)


    def test_span_label(self):
        sample = self.utils.create_and_label_sample()
        self.assertTrue("span_id" in sample.df.columns)
        self.assertTrue("label" in sample.df.columns)
        self.assertTrue("link" in sample.df.columns)
        self.assertTrue("link_label" in sample.df.columns)
        self.assertTrue("seg_id" in sample.df.columns)
        self.assertTrue("target_id" in sample.df.columns)
        self.assertTrue("am_id" in sample.df.columns)
        self.assertTrue("adu_id" in sample.df.columns)
 

    def test_sample_df(self):
        sample = self.utils.create_and_label_sample()
        gold_df = self.utils.load_gold_df()
        self.assertTrue(gold_df.equals(sample.df))


    def test_sample_str(self):
        sample = self.utils.create_sample()
        correct_str = "this is a paragrap which is also a segments . this is a not a segment . however , here starts a segement . this , right after this comma , is also a segment ."
        self.assertEqual(str(sample), correct_str)


    def test_token_length(self):
        sample = self.utils.create_sample()
        length = sample.get("token", "length")
        self.assertIsInstance(length, int)
        self.assertEqual(length, 36)


    def test_seg_length(self):
        sample = self.utils.create_and_label_sample()
        length = sample.get("seg", "length")
        self.assertIsInstance(length, int)
        self.assertEqual(length, 3)


    def test_seg_span_idxs(self):
        sample = self.utils.create_and_label_sample()
        span_idxs = sample.get("seg","span_idxs")
        gold_span_idxs = torch.LongTensor([[ 0,  9],
                                            [17, 23],
                                            [31, 35]])

        self.assertTrue(torch.is_tensor(span_idxs))
        self.assertTrue(torch.equal(span_idxs, gold_span_idxs))


    def test_am_span_idxs(self):
        sample = self.utils.create_and_label_sample()
        span_idxs = sample.get("am","span_idxs")
        gold_span_idxs = torch.LongTensor([[ 0,  0],
                                            [ 0,  0],
                                            [24, 30]])
        self.assertTrue(torch.is_tensor(span_idxs))
        self.assertTrue(torch.equal(span_idxs, gold_span_idxs))


    def test_token_get(self):
        sample = self.utils.create_and_label_sample()
        gold_df = self.utils.load_gold_df()

        for c in gold_df.columns:
            values = sample.get("token", c)

            if c in ["str", "pos", "deprel"]:
                self.assertIsInstance(values, list)
                self.assertTrue(np.array_equal(values, gold_df[c].to_list()))
            else:
                self.assertTrue(torch.is_tensor(values))
                self.assertTrue(torch.equal(values, torch.LongTensor(gold_df[c].to_numpy())))

           
    def test_split(self):
        sample = self.utils.create_and_label_sample()
        list_samples = sample.split("paragraph")
        self.assertIsInstance(list_samples, list)
        self.assertEqual(len(list_samples), 2)

        para_strings = [
                            "this is a paragrap which is also a segments .",
                            "this is a not a segment . however , here starts a segement . this , right after this comma , is also a segment ."
                            ]

        para_links = [
                            torch.LongTensor([0]),
                            torch.LongTensor([1,1]),
                            ]


        zipped = zip(list_samples, para_strings, para_links)

        # we test  the string and if the links have been normalized to be within segments
        for sample, string, links in zipped:
            self.assertEqual(str(sample), string)
            self.assertTrue(torch.equal(sample.get("seg","link"), links))


    def _raise_get_error(self, error, level, key):
        sample  = self.utils.create_and_label_sample()
        with self.assertRaises(error):
            sample.get(level, key)
    

    def test_get_seg_str(self):
        self._raise_get_error(KeyError, "seg", "str")


    def test_get_seg_pos(self):
        self._raise_get_error(KeyError, "seg", "pos")


    def test_get_seg_deprel(self):
        self._raise_get_error(KeyError, "seg", "deprel")


    def test_get_random_keys_levels(self):
        self._raise_get_error(KeyError, "lollll", "label")
        self._raise_get_error(KeyError, 10, "label")
        self._raise_get_error(KeyError, "hello", 20)


    def test_copy_sample(self):
        sample  = self.utils.create_and_label_sample()
        sample_copy = sample.copy(clean=True)
        self.assertFalse(sample.df.equals(sample_copy.df))


    def test_add_seg_label(self):
        sample = self.utils.create_and_label_sample()
        sample_pred = sample.copy(clean=True)

        # using the gold labels
        token_bio = sample.get("token", "seg")
        seg_links = sample.get("seg", "link")

        # do segmentation, so we can add labels to segments
        sample_pred.add("token", "seg", token_bio)

        # add links to segments
        sample_pred.add("seg", "link", seg_links)

        # make sure we have the same labels we added
        is_equal = torch.equal(seg_links, sample_pred.get("seg", "link"))
        self.assertTrue(is_equal)



    def test_add_seg_labels_inb4_segmentation(self):
        sample = self.utils.create_and_label_sample()
        sample_pred = sample.copy(clean=True)

        # using the gold labels
        seg_links = sample.get("seg", "link")

        # as we have removed all previous labels, this include segmentation information
        # hence we cannot add anything to segment level yet, we first need to do segmentaion
        with self.assertRaises(RuntimeError):
            sample_pred.add("seg", "link", seg_links)



    def test_add_decouple_seg(self):
        sample  = self.utils.create_and_label_sample()
        sample_pred = sample.copy(clean=True)

        # using the gold labels
        seg_label_preds = sample.get("token", "seg+label")

        # label pred with true labels
        sample_pred.add("token", "seg+label", seg_label_preds)

        # make sure we have the same labels we added
        is_equal = torch.equal(seg_label_preds, sample_pred.get("token", "seg+label"))
        self.assertTrue(is_equal)
