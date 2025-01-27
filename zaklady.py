from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

def rgb2gs(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140]).astype(np.uint8)

def threshold(im, t):
    return (im > t).astype(np.uint8)

def threshold2(im, t):
    im2 = im.copy()
    im2[im2 > t] = 255
    return im2

def image_histogram_equalization(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()
    cdf = (number_bins-1) * cdf / cdf[-1] 
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(image.shape).astype(np.uint8)

im = imread('img/gbv.jpg')
im_gs =rgb2gs(im)
#im_gs2 = threshold2(im_gs * 1.5, 255)
im_gs2 = image_histogram_equalization(im_gs, 256)
plt.figure(figsize= (15, 5))
plt.subplot(2, 2, 1)
plt.imshow(im_gs, cmap=plt.cm.gray)
plt.subplot(2, 2, 2)
plt.imshow(im_gs2, cmap=plt.cm.gray)
plt.subplot(2, 2, 3)
plt.hist(im_gs.flatten(), range=(0, 255), bins=256)
plt.subplot(2, 2, 4)
plt.hist(im_gs2.flatten(), range=(0, 255), bins=256)
plt.show()