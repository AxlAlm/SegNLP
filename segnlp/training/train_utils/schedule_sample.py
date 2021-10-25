
# basics
from math import exp, floor
from random import random


class ScheduleSampling():
    """ Scheduled Sampling: A schedule mechanism that randomly decides whether
    to use the ground truth or the predicted variable in the current training
    epoch. Dependeing on the schedule, a decayed probaility epsilon is
    calculated as a function of current epoch number.

    Reference: https://arxiv.org/pdf/1506.03099.pdf

    Args
    schedule:   str, schedule mechanism
                either:
                `linear`: Linear decay mechanism,
                `exponential`: Exponential decay mechanism or
                `inverse_sig`: Inverse sigmoid decay mechanism
    k:          float, a constant to calculate epsilon. It determine the slope
                of decaying.
                k < 1 in Exponential decay mechanism
                k > 1 in Inverse sigmoid decay mechanism
    c:          float, a constant that determines the offset of the linear
                decay mechanism. Not needed for the others schedule mechanisms.
    e:          float, a constant required for the linear decay mechanism,

    return:
                None
    """
    def __init__(self,
                 k: float,
                 schedule: str = "linear",
                 c: float = 1e-3,
                 e: float = 0.0):

        # TODO Assert conditions of k and others
        if schedule == "linear":
            epsilon = lambda epoch: max(e, k - c * epoch)  # noqa: E731

        elif schedule == "exponential":
            epsilon = lambda epoch: k**epoch  # noqa: E731

        elif schedule == "inverse_sig":
            epsilon = lambda epoch: k / (k + exp(epoch / k))  # noqa: E731

        else:
            raise KeyError(f'"{schedule}" is not a supported schedule')

        self.calc_sampling_prob = epsilon

    def __call__(self, epoch):
        epsilon = self.calc_sampling_prob(epoch)
        coin_flip = floor(random() * 10) / 10

        return bool(epsilon >= coin_flip)


