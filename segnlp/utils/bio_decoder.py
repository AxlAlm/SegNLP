#basics
import re 
import numpy as np
from typing import List


#segnlp
from .array import ensure_numpy
from .array import ensure_flat
from .array import create_mask

from segnlp import utils


class BIODecoder:

    def __init__(self, 
                B:list=["B"],
                I:list=["I"],
                O:list=["O"],
                apply_correction:bool = True
                ):

        Bs = "-|".join([str(b) for b in B]) + "-"
        Is = "-|".join([str(i) for i in I]) + "-"
        Os = "-|".join([str(o) for o in O]) + "-"
        
        # if we have invalid BIO structure we can correct these
        #  https://arxiv.org/pdf/1704.06104.pdf, appendix
        # 1) I follows 0 -> allow OI to be interpreted as a B
        if apply_correction:
            Bs += f"|(?<=({Os}))({Is})|<START>-"

        self.pattern = re.compile(f"(?P<UNIT>({Bs})({Is})*)|(?P<NONE>({Os})+)")
        

    def __call__(self, encoded_bios:List[str], seg_id_start:int = 0):

        encoded_bios = ensure_numpy(encoded_bios)
        encoded_bios_str = "<START>-" + "-".join(encoded_bios.astype(str)) + "-"
        all_matches = re.finditer(self.pattern, encoded_bios_str)

        tok_seg_ids = []
        i = seg_id_start
        for m in all_matches:

            match_string = m.group(0).replace("<START>-", "")
            length = len(match_string.split("-")[:-1])

            if length == 0:
                continue

            if m.groupdict()["UNIT"] is None:
                tok_seg_ids.extend([None] * length)
            else:
                tok_seg_ids.extend([i] * length)
                i += 1

        return tok_seg_ids

        #ids = 



        # return {
        #             "span":{
        #                     "ids": np.array(seg_ids),
        #                     "none_span_mask":none_span_mask,
        #                     "tok_lengths": np.array(span_lengths),
        #                     "start": span_starts,
        #                     "end": span_ends,
        #                     "idxs": span_indexes
        #                     },
        #             "unit":{
        #                     "none_span_mask": np.array(none_span_mask, dtype=bool),
        #                     "tok_lengths": p.array(span_lengths),
        #                     "ids": np.array(seg_ids)[none_span_mask],
        #                     "start": span_starts[none_span_mask],
        #                     "end": span_ends[none_span_mask],
        #                     "idxs": span_indexes[none_span_mask]
        #                     }

        #             }, i

    # def __call__(
    #             self,
    #             batch_encoded_bios:np.ndarray, 
    #             lengths:np.ndarray, 
    #             ):
    
    #     batch_size = batch_encoded_bios.shape[0]

    #     bio_data = {
    #                 "span":{
    #                         "lengths_tok":[],
    #                         "lengths": [],
    #                         "span_idxs": [],
    #                         "none_span_mask":[],
    #                         "start":[],
    #                         "end":[],
    #                         },
    #                 "seg":{
    #                         "lengths_tok":[],
    #                         "lengths":[],
    #                         "span_idxs": [],
    #                         "none_span_mask":[],
    #                         "start":[],
    #                         "end":[],
    #                         },
    #                 "max_segs":0
    #                 }


    #     for i in range(batch_size):
    #         span_lengths, none_span_mask, seg_ids = self.__decode_sample(
    #                                                             encoded_bios=batch_encoded_bios[i][:lengths[i]]
    #                                                             )

    #         span_ends = np.cumsum(span_lengths)
    #         span_starts = np.insert(span_ends,0,0)[:-1]
    #         span_indexes = np.stack((span_starts, span_ends), axis=-1)
    #         seg_indexes = span_indexes[none_span_mask]
    #         seg_lengths = span_lengths[none_span_mask]

    #         # bio_data["span"]["lengths"].append(len(span_lengths))
    #         # bio_data["span"]["lengths_tok"].append(span_lengths.tolist())

    #         # bio_data["span"]["none_span_mask"].append(none_span_mask.tolist())
    #         # bio_data["span"]["span_idxs"].append(span_indexes.tolist())
    #         # bio_data["span"]["start"].append(span_starts.tolist())
    #         # bio_data["span"]["end"].append(span_ends.tolist())
            
    #         # seg_length = sum(none_span_mask)
    #         # bio_data["seg"]["lengths"].append(seg_length)
    #         # bio_data["seg"]["lengths_tok"].append(span_lengths[none_span_mask].tolist())
    #         # bio_data["seg"]["span_idxs"].append(span_indexes[none_span_mask].tolist())
    #         # bio_data["seg"]["start"].append(span_starts[none_span_mask].tolist())
    #         # bio_data["seg"]["end"].append(span_ends[none_span_mask].tolist())

    #         # bio_data["max_segs"] = max(seg_length, bio_data["max_segs"])

    #     bio_data["seg"]["mask"] = create_mask(bio_data["seg"]["lengths"])
  
    #     return bio_data
