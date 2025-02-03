from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from math import radians
from common import rgb2gs


def rotate_translate(src_img,angle_deg, pivot_point):

    angle = radians(angle_deg)

    rotation_mat = np.array([[np.cos(angle),-np.sin(angle)],
                            [np.sin(angle),np.cos(angle)]])
    height, width = src_img.shape
       
    new_img = np.zeros(src_img.shape, dtype=np.uint8) 

    for y in range(height):
        for x in range(width):
            xy_mat = np.array([[x-pivot_point[0]], [y-pivot_point[1]]])
            
            rotate_mat = np.dot(rotation_mat,xy_mat)

            old_x = pivot_point[0] + round(rotate_mat[0][0])
            old_y = pivot_point[1] + round(rotate_mat[1][0])


            if (0 <= old_x < width) and (0 <= old_y < height): 
                new_img[y,x] = src_img[old_y, old_x]

    return new_img

def rotate(src_img,angle_deg):

    angle = radians(angle_deg)

    rotation_mat = np.array([[np.cos(angle),-np.sin(angle)],
                            [np.sin(angle),np.cos(angle)]])
    height, width = src_img.shape
       
    new_img = np.zeros(src_img.shape, dtype=np.uint8) 

    for y in range(height):
        for x in range(width):
            xy_mat = np.array([[x], [y]])
            
            rotate_mat = np.dot(rotation_mat, xy_mat)

            old_x = round(rotate_mat[0][0])
            old_y = round(rotate_mat[1][0])


            if (0 <= old_x < width) and (0 <= old_y < height): 
                new_img[y, x] = src_img[old_y, old_x]

    return new_img

im = imread('img/gbv.jpg')
im =rgb2gs(im)


im90 = np.rot90(im)

im_rot = rotate(im, 45)
im_rot2 = rotate_translate(im, 45, [im.shape[1]//2, im.shape[0]//2])

plt.figure(figsize= (15, 5))
plt.subplot(2, 2, 1)
plt.imshow(im, cmap=plt.cm.gray)
plt.subplot(2, 2, 2)
plt.imshow(im90, cmap=plt.cm.gray)
plt.subplot(2, 2, 3)
plt.imshow(im_rot, cmap=plt.cm.gray)
plt.subplot(2, 2, 4)
plt.imshow(im_rot2, cmap=plt.cm.gray)
plt.show()