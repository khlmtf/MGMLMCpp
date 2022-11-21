import numpy as np

def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res
