
#basics
import re

#segnlp
from .ams import ams

#just a fuzzy match for commas
sorted_ams = sorted(ams, key= lambda x:len(x), reverse=True)
cond = lambda x: x[-2] == " ," or x[-1] == ","
am_pattern = re.compile("(?<=START)"+"|(?<=START)".join([f"{a[:-2]}(?:\s,|,|)" if cond(a) else a for a in sorted_ams]))

def find_am(list_tokens:list):
    """[summary]

    Parameters
    ----------
    list_tokens : list
        list of tokens

    Returns
    -------
    tuple(list,list)
        matches tokens,
        token indexes
    """
    
    string = "START"+" ".join(list_tokens).lower()
    match = re.findall(am_pattern, string)

    if match:
        matched_tokens = match[0].split()
        indexes = list(range(len(matched_tokens)))
        return matched_tokens, indexes
    else:
        return [], []

    return match
