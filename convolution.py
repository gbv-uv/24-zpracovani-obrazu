import numpy as np
from common import rgb2gs
from matplotlib.image import imread
import matplotlib.pyplot as plt
from scipy import ndimage

im = imread('img/gbv_s.jpg')
im = rgb2gs(im)
im = im.astype('int32')

edge_vertical = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
    ])


edge_horizontal = np.array([
    [-1,-1,-1],
    [ 0, 0, 0],
    [ 1, 1, 1]
    ])


blur = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    ])

kernel = blur

result = ndimage.convolve(im, kernel, mode='reflect')

plt.figure(figsize= (15, 5))
plt.subplot(1, 2, 1)
plt.imshow(im, cmap=plt.cm.gray)
plt.subplot(1, 2, 2)
plt.imshow(result, cmap=plt.cm.gray)
plt.show()