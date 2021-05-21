

#basics
from typing import List, Dict, Tuple, Union
import numpy as np
from scipy import stats


#deepsig
from deepsig import aso

#segnlp
import segnlp.utils as utils


class StatSig:


    def __stat_sig(self, a_dist:List, b_dist:List, ss_test="aso"):
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
        
        is_sig = False
        if ss_test == "aso":
            v = aso(a_dist, b_dist)
            is_sig = v <= 0.5

        elif ss_test == "mwu":
            v = stats.mannwhitneyu(a_dist, b_dist, alternative='two-sided')
            is_sig = v <= 0.05

        else:
            raise RuntimeError(f"'{ss_test}' is not a supported statistical significance test. Choose between ['aso', 'mwu']")

        return is_sig, v


    def model_comparison(self, a_dist:List, b_dist:List, ss_test="aso"):
        """

        This function compares two approaches --lets call these A and B-- by comparing their score
        distributions over n number of seeds.

        first we need to figure out the proability that A will produce a higher scoring model than B. Lets call this P.
        If P is higher than 0.5 we cna say that A is better than B, BUT only if P is significantly different from 0.5. 
        To figure out if P is significantly different from 0.5 we apply a significance test.

        https://www.aclweb.org/anthology/P19-1266.pdf
        https://export.arxiv.org/pdf/1803.09578

        """

        a_dist = utils.ensure_numpy(a_dist)
        b_dist = utils.ensure_numpy(b_dist)

        if all(np.sort(a_dist) == np.sort(b_dist)):
            return False, 0.0, 1.0

        larger_than = a_dist >= b_dist
        x = larger_than == True
        prob = sum(larger_than == True) / larger_than.shape[0]

        a_better_than_b = None
        v = None
        if prob > 0.5:
            
            is_sig, v = self.__stat_sig(a_dist, b_dist, ss_test=ss_test)

            if is_sig:
                a_better_than_b = True

        return a_better_than_b, prob, v

    