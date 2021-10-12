

#basics
from typing import List, Dict, Sequence, Tuple, Union
import numpy as np
from scipy import stats


#deepsig
from deepsig import aso

#segnlp
from .array import ensure_numpy


def statistical_significance_test(a:np.ndarray, b:np.ndarray, ss_test="aso"):
    """
    Tests if there is a significant difference between two distributions. Normal distribtion not needed.
    Two tests are supported. We prefer 1) (see https://www.aclweb.org/anthology/P19-1266.pdf)

    :

        1) Almost Stochastic Order

            Null-hypothesis:
                H0 : aso-value >= 0.5
            
            i.e. ASO is not a p-value and instead the threshold is different. We want our score to be
            below 0.5, the lower it is the more sure we can be that A is better than B.    


        2) Mann-Whitney U 

            Null-hypothesis:

                H0: P is not significantly different from 0.5
                HA: P is significantly different from 0.5
            
            p-value >= .05


    1) is prefered

    """
    
    if ss_test == "aso":
        v = aso(a, b)
        return v <= 0.5, v

    elif ss_test == "mwu":
        v = stats.mannwhitneyu(a, b, alternative='two-sided')
        return v <= 0.05, v

    else:
        raise RuntimeError(f"'{ss_test}' is not a supported statistical significance test. Choose between ['aso', 'mwu']")




def compare_dists(a:Sequence, b:Sequence, ss_test="aso"):
    """

    This function compares two approaches --lets call these A and B-- by comparing their score
    distributions over n number of seeds.

    first we need to figure out the proability that A will produce a higher scoring model than B. Lets call this P.
    If P is higher than 0.5 we cna say that A is better than B, BUT only if P is significantly different from 0.5. 
    To figure out if P is significantly different from 0.5 we apply a significance test.

    https://www.aclweb.org/anthology/P19-1266.pdf
    https://export.arxiv.org/pdf/1803.09578

    """

    a = np.sort(ensure_numpy(a))[::-1]
    b = np.sort(ensure_numpy(b))[::-1]

    if np.array_equal(a, b):
        return False, 0.0

    better_then = a >= b
    prob = sum(better_then) / len(better_then)

    v = 0.0
    is_better = False
    if prob > 0.5:

        is_better, v = statistical_significance_test(a, b, ss_test=ss_test)

    return is_better, v

