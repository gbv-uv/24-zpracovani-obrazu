import numpy as np

def rgb2gs(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140]).astype(np.uint8)