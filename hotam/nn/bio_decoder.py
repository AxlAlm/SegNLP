
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

        #self.pattern = re.compile(f"({Bs})({Is})*|({Os})+")
        self.pattern = re.compile(f"(?P<AC>({Bs})({Is})*)|(?P<NONE>({Os})+)")
        

    def _bio_decode_sample(self, encoded_bios):

        encoded_bios = ensure_numpy(encoded_bios)
        encoded_bios_str = "-".join(encoded_bios.astype(str)) + "-"

        all_matches = re.finditer(self.pattern, encoded_bios_str)

        seg_types = []
        seg_lengths = []
        ac_length = 0
        for m in all_matches:
            length = len(string.split("-")[:-1])
            seg_type = None if groupdict["AC"] is None else "AC"

            seg_lengths.append(length)
            seg_type.append(seg_type)
            
            if seg_type == "AC":
                ac_length += 1 

        return seg_lengths, seg_types, ac_length


    def decode(self, batch_encoded_bios:np.ndarray, lengths:np.ndarray):
        
        batch_size = batch_encoded_bios.shape[0]

        # sample = [2,3,10,6] where each number indicate the lenght of a Argument Component 
        sample_seg_lengths = []

        # sample = [AC,NONE,AC,AC] were AC indicate that segment is a argument component, NONE that its not an Argument Component
        sample_seg_types = []

        # sample = 3, nr of predicted Argument Components in the sample
        sample_ac_lengths = []

        for i in range(batch_size):
            seg_lengths, seg_types, ac_length = self._bio_decode_sample(
                                                                    encoded_bios=batch_encoded_bios[i][:lengths[i]]
                                                                    )
            sample_seg_lengths.append(seg_lengths)
            sample_seg_types.append(seg_types)
            sample_ac_lengths.append(ac_length)

        return sample_seg_lengths, sample_seg_types, sample_ac_lengths
