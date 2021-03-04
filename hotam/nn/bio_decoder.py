
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
        self.pattern = re.compile(f"(?P<UNIT>({Bs})({Is})*)|(?P<NONE>({Os})+)")
        

    def _bio_decode_sample(self, encoded_bios):

        encoded_bios = ensure_numpy(encoded_bios)
        encoded_bios_str = "-".join(encoded_bios.astype(str)) + "-"
        all_matches = re.finditer(self.pattern, encoded_bios_str)

        span_types = []
        span_lengths = []
        nr_units = 0
        for m in all_matches:
            length = len(string.split("-")[:-1])
            span_type = 0 if groupdict["UNIT"] is None else 1

            span_lengths.append(length)
            span_types.append(span_type)
            
            if span_type == "UNIT":
                nr_units += 1 

        return span_lengths, span_types, nr_units


    def decode(self, batch_encoded_bios:np.ndarray, lengths:np.ndarray):
        
        batch_size = batch_encoded_bios.shape[0]

        # sample = [2,3,10,6] where each number indicate the lenght of a spans
        sample_span_lengths = []

        # sample = [0,1,0,1,0,1,1] mask which tells us which spans are unit and whihc are not
        sample_span_types = []

        # sample = 3, nr of predicted units in the sample
        unit_lengths = []

        for i in range(batch_size):
            span_lengths, span_types, nr_units = self._bio_decode_sample(
                                                                    encoded_bios=batch_encoded_bios[i][:lengths[i]]
                                                                    )
            sample_span_lengths.append(span_lengths)
            sample_span_types.append(span_types)
            unit_lengths.append(nr_units)

        return sample_span_lengths, sample_span_types,  unit_lengths
