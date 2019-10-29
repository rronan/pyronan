import numpy as np


class RollingStatistic(object):
    def __init__(self, window_size, average, variance):
        self.N = window_size
        self.average = average
        self.variance = variance
        self.stddev = np.sqrt(variance)

    def update(self, new, old):
        oldavg = self.average
        newavg = oldavg + (new - old) / self.N
        self.average = newavg
        self.variance += (new - old) * (new - newavg + old - oldavg) / (self.N - 1)
        self.stddev = np.sqrt(self.variance)
