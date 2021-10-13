#basics
import re 
import numpy as np
from typing import List

#segnlp
from .array import ensure_numpy

class BIODecoder:

    def __init__(self, 
                O:int = 0,
                B:int = 1,
                I:int = 2,
                apply_correction:bool = True,
                start_value = 9999
                ):

        self.start_value = f"{start_value}-"
        O = f"{O}-"
        B = f"{B}-"
        I = f"{I}-"

        # if we have invalid BIO structure we can correct these
        #  https://arxiv.org/pdf/1704.06104.pdf, appendix
        # 1) I follows 0 -> allow OI to be interpreted as a B
        if apply_correction:
            B = f"{B}|(?<={O}){I}|{self.start_value}{I}"
            
        self.pattern = re.compile(f"(?P<SEG>({B})({I})*)|(?P<NONE>({O})+|({I})+)")
        
        
    def __call__(self, encoded_bios:List[str], sample_start_idxs:np.ndarray=None):

        encoded_bios = ensure_numpy(encoded_bios).astype(int)

        if sample_start_idxs is not None:
            # we insert an arbitrary value 9999 to be used as a start pattern so we can seperate samples
            # 9999 is the default value for all SegDecoders
            seg_preds_w_starts = np.insert(encoded_bios, sample_start_idxs, 9999)
            encoded_bios_str = "-".join(seg_preds_w_starts.astype(str)) + "-"

        else:
            encoded_bios_str = self.start_value + "-".join(encoded_bios.astype(str)) + "-"

        all_matches = re.finditer(self.pattern, encoded_bios_str)

        tok_seg_ids = np.full(encoded_bios.shape, fill_value=np.nan)
        seg_id = 0
        p = 0
        for m in all_matches:
            
            match_string = m.group(0).replace(self.start_value, "")
            length = len(match_string.split("-")[:-1])
            
            if length == 0:
                continue

            if not m.groupdict()["SEG"] is None:
        
                tok_seg_ids[p:p+length] = seg_id
                seg_id += 1
            
            p += length

        return tok_seg_ids