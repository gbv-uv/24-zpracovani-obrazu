from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

black = 0
white = 255

height = 600
width = 800

image = np.zeros(shape=(height, width), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        if x % 2 == 0 and y % 2 == 0:
            image[y, x] = white



plt.figure(figsize= (15, 12))
plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.show()