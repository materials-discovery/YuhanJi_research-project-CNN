import numpy as np

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range