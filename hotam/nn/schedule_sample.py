from math import exp


class ScheduleSampling():
    def __init__(self,
                 schedule: str = "linear",
                 k: float = 1.0,
                 c: float = 1e-3,
                 e: float = 0.0):

        self.step = 0
        self.k = k
        self.c = c
        self.e = e
        self.schedule = schedule

    def calc_epsilon(self):
        if self.schedule == "linear":
            epsilon = max(self.e, self.k - self.c * self.step)

        elif self.schedule == "exponential":
            epsilon = self.k**self.step

        elif self.schedule == "inverse_sig":
            epsilon = self.k / (self.k + exp(self.step / self.k))

        else:
            # error
            pass

        return epsilon

    def next(self):
        prob = self.calc_epsilon()
        self.step += 1
        return prob
