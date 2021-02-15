
from hotam.utils import ensure_numpy, ensure_flat

import re 
import torch
import numpy as np
#torch.nn.Module

class BIO_Decoder():


    def __init__(self, B:list, I:list, O:list, apply_correction:bool=True):
        Bs = "-|".join([str(b) for b in B]) + "-"
        Is = "-|".join([str(i) for i in I]) + "-"
        Os = "-|".join([str(o) for o in O]) + "-"
        self._Os = O

        self._apply_correction = apply_correction

        # if we have invalid BIO structure we can correct these
        #  https://arxiv.org/pdf/1704.06104.pdf, appendix
        # 1) I follows 0 -> allow OI to be interpreted as a B
        if self._apply_correction:
            Bs += f"|(?<=({Os}))({Is})"

        self.pattern = re.compile(f"({Bs})({Is})*|({Os})+")


    def _bio_decode_sample(self, encoded_bios):

        encoded_bios = ensure_flat(ensure_numpy(encoded_bios))
        encoded_bios_str = "-".join(encoded_bios.astype(str)) + "-"

        self.__lengths = []
        self.__seg_types = []
        def repl(m):
            bio_list = m.group().split("-")[:-1] #when splitting on "-" we will alway create an empty "" at the end
            length = len(bio_list)

            seg_type = "AC"
            set_labels = list(set(bio_list))
            if int(set_labels[0]) in self._Os:
                seg_type = None
      
            self.__lengths.append(length)
            self.__seg_types.append(seg_type)
  
            return ""

        re.sub(self.pattern, repl, encoded_bios_str)


    def decode(self, batch_bios, sample_lengths):
        
        batch_size = batch_bios.shape[0]
        sample_seg_lengths = []
        sample_seg_types = []
        for i in range(batch_size):
            self._bio_decode_sample(batch_bios[i][:sample_lengths[i]])
            sample_seg_lengths.append(self.__lengths)
            sample_seg_types.append(self.__lengths)

        return sample_seg_lengths, sample_seg_types
