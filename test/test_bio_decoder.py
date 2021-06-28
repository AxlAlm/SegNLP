
#basics
import unittest
import numpy as np

#segnlp
from segnlp.utils import BIODecoder


class TestBIODecoder(unittest.TestCase):

    def test_decoding(self):
        
        bio_decoder = BIODecoder()
        seg_ids = bio_decoder([2,2,2,1,2,2,2,0,0,0,0,1,1,2,2,2])

        self.assertEqual(seg_ids, np.array([0,0,0,1,1,1,1,None,None,None,None,2,3,3,3,3]))

  
    def test_decoding_wo_correction(self):

        bio_decoder = BIODecoder(apply_correction=False)
        seg_ids = bio_decoder([2,2,2,1,2,2,2,0,0,0,0,1,1,2,2,2])

        self.assertEqual(seg_ids, np.array([None,None,None,1,1,1,1,None,None,None,None,2,3,3,3,3]))

