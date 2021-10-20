#basics
import re 
import numpy as np
from typing import List

#segnlp
from .array import ensure_numpy

        
def decode_bio(
                encoded_bios:List[str], 
                apply_correction : bool = True, 
                B : int = 1, 
                O : int = 0 , 
                I : int = 2,
                ):

    start_value = "9999-"
    O = f"{O}-"
    B = f"{B}-"
    I = f"{I}-"

    # if we have invalid BIO structure we can correct these
    #  https://arxiv.org/pdf/1704.06104.pdf, appendix
    # 1) I follows 0 -> allow OI to be interpreted as a B
    if apply_correction:
        B = f"{B}|(?<={O}){I}|{start_value}{I}"
    
    # setup a pattern
    pattern = re.compile(f"(?P<SEG>({B})({I})*)|(?P<NONE>({O})+|({I})+)")
    
    #ensure the numbers are ints and a numpy
    encoded_bios = ensure_numpy(encoded_bios).astype(int)

    # turn the BIO labels into string
    encoded_bios_str = f'{start_value}{"-".join(encoded_bios.astype(str))}-'

    # find matches
    all_matches = re.finditer(pattern, encoded_bios_str)

    tok_seg_ids = np.full(encoded_bios.shape, fill_value=-1)
    seg_id = 0
    p = 0
    for m in all_matches:
        
        match_string = m.group(0).replace(start_value, "")
        length = len(match_string.split("-")[:-1])
        
        if length == 0:
            continue

        if not m.groupdict()["SEG"] is None:
    
            tok_seg_ids[p:p+length] = seg_id
            seg_id += 1
        
        p += length

    return tok_seg_ids