import numpy as np 
from scipy.special import erfinv


def uniform(sample, lo=-1, hi=1):
    return lo + (hi - lo) * sample

def normal(sample, avg=0.0, std=1.0):
    return avg + std * np.sqrt(2) * erfinv(2 * sample - 1)

def lognormal(sample, avg=0.0, std=1.0):
    return np.exp(normal(sample, avg, std))

def bernoulli(sample, p=0.5):
    return (sample > p)
