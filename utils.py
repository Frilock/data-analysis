import numpy as np


def prepare_x(x):
    x_std = x.std(axis=0)
    x_mean = x.mean(axis=0)
    x = (x - x_mean) / x_std
    x = np.concatenate((np.ones(len(x))[:, np.newaxis], x), axis=1)
    return x
