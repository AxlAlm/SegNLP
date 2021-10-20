
#basics
import unittest
import pandas as pd
import numpy as np

#pytroch
import torch

# segnlp
from utils import Utils
from segnlp.data import Batch


class TestSample(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.utils = Utils() 


    def test_create_batch(self):
        batch  = self.utils.create_batch(size = 10)
        self.assertIsInstance(batch, Batch)


    def test_length(self):
        batch = self.utils.create_batch(size = 10)
        self.assertEqual(len(batch), 10)


    def _get_tensor(self, level, key, gold_data):
        gold_data = torch.LongTensor([gold_data]*10)
        batch  = self.utils.create_batch(size = 10)
        data = batch.get(level, key)
        self.assertTrue(torch.is_tensor(data))
        self.assertTrue(torch.equal(data, gold_data))
    

    def test_get_seg_link(self):
        self._get_tensor("seg", "link", [0, 2, 2])


    def test_get_seg_label(self):
        self._get_tensor("seg", "label", [1, 1, 2])


    def test_get_seg_label(self):
        self._get_tensor("seg", "link_label", [0, 1, 0])


    def test_get_token_seg(self):
        gold_data = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2]
        self._get_tensor("token", "seg", gold_data)
    

    def _get_string_values(self, level, key, gold_data):
        batch  = self.utils.create_batch(size = 10)
        data = batch.get(level, key)
        gold_data = [gold_data]*10
        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], list)
        self.assertIsInstance(data[0][0], str) # no need for this
        self.assertEqual(data, gold_data)


    def test_get_token_str(self):
        gold_data = ['this', 'is', 'a', 'paragrap', 'which', 'is', 'also', 'a',
                    'segments', '.', 'this', 'is', 'a', 'not', 'a', 'segment', '.',
                    'however', ',', 'here', 'starts', 'a', 'segement', '.', 'this',
                    ',', 'right', 'after', 'this', 'comma', ',', 'is', 'also', 'a',
                    'segment', '.']
        self._get_string_values("token", "str", gold_data)


    def test_get_token_pos(self):
        gold_data = ['DT', 'VBZ', 'DT', 'NN', 'WDT', 'VBZ', 'RB', 'DT', 'NNS', '.',
                    'DT', 'VBZ', 'DT', 'RB', 'DT', 'NN', '.', 'RB', ',', 'RB', 'VBZ',
                    'DT', 'NN', '.', 'DT', ',', 'RB', 'IN', 'DT', 'NN', ',', 'VBZ',
                    'RB', 'DT', 'NN', '.']
        self._get_string_values("token", "pos", gold_data)


    def test_get_token_deprel(self):
        gold_data = ['nsubj', 'ROOT', 'det', 'attr', 'nsubj', 'relcl', 'advmod', 'det',
        'attr', 'punct', 'nsubj', 'ROOT', 'det', 'neg', 'det', 'attr',
        'punct', 'advmod', 'punct', 'advmod', 'ROOT', 'det', 'dobj',
        'punct', 'intj', 'punct', 'advmod', 'prep', 'det', 'pobj', 'punct',
        'ROOT', 'advmod', 'det', 'attr', 'punct']
        self._get_string_values("token", "deprel", gold_data)


    def test_get_pred(self):
        pass


    def test_add(self):
        pass