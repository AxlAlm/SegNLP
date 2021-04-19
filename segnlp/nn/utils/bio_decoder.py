

import re 
import torch
import numpy as np

#segnlp
from . import ensure_numpy, ensure_flat


def bio_decode(
                batch_encoded_bios:np.ndarray, 
                lengths:np.ndarray, 
                apply_correction=True,
                B:list=["B"],
                I:list=["I"],
                O:list=["O"],
                ):
        
    def _bio_decode_sample(pattern, encoded_bios):

        encoded_bios = ensure_numpy(encoded_bios)
        encoded_bios_str = "<START>-" + "-".join(encoded_bios.astype(str)) + "-"
        all_matches = re.finditer(pattern, encoded_bios_str)

        span_types = []
        span_lengths = []
        nr_units = 0
        for m in all_matches:

            match_string = m.group(0).replace("<START>-", "")
            length = len(match_string.split("-")[:-1])

            if length == 0:
                continue

            span_type = 0 if m.groupdict()["UNIT"] is None else 1

            span_lengths.append(length)
            span_types.append(span_type)
            
            if span_type:
                nr_units += 1 

        return span_lengths, span_types, nr_units


    Bs = "-|".join([str(b) for b in B]) + "-"
    Is = "-|".join([str(i) for i in I]) + "-"
    Os = "-|".join([str(o) for o in O]) + "-"
    
    # if we have invalid BIO structure we can correct these
    #  https://arxiv.org/pdf/1704.06104.pdf, appendix
    # 1) I follows 0 -> allow OI to be interpreted as a B
    if apply_correction:
        Bs += f"|(?<=({Os}))({Is})|<START>-"

    #self.pattern = re.compile(f"({Bs})({Is})*|({Os})+")
    pattern = re.compile(f"(?P<UNIT>({Bs})({Is})*)|(?P<NONE>({Os})+)")
    
    batch_size = batch_encoded_bios.shape[0]

    # sample = [2,3,10,6] where each number indicate the lenght of a spans
    sample_span_lengths = []

    # sample = [0,1,0,1,0,1,1] mask which tells us which spans are unit and whihc are not
    sample_span_types = []

    # sample = 3, nr of predicted units in the sample
    unit_lengths = []

    for i in range(batch_size):
        span_lengths, span_types, nr_units = _bio_decode_sample(
                                                                pattern,
                                                                encoded_bios=batch_encoded_bios[i][:lengths[i]]
                                                                )
        sample_span_lengths.append(span_lengths)
        sample_span_types.append(span_types)
        unit_lengths.append(nr_units)

    return sample_span_lengths, sample_span_types,  unit_lengths
