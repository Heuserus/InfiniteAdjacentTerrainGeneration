import os
import random
import numpy as np
import png
import numba.cuda as cuda
import math

from PIL import Image, ImageOps
#from scipy.misc import imread
from numpy import asarray
from skimage.color import rgb2gray, gray2rgb
import cv2

@cuda.jit
def autocontrast_kernel(img,im_terrarium):
    
    h, w = cuda.grid(2)
    
    if h < img.shape[0] and w < img.shape[1]:
        gray_value = img[h][w]
        im_terrarium[h][w][0] = math.floor(gray_value // 32)
        im_terrarium[h][w][1] = gray_value & 0xFF

def autocontrast(img,im_terrarium):

    img_gpu = cuda.to_device(img)
    im_terrarium_gpu = cuda.to_device(im_terrarium)
    
    threads_per_block = (16, 16)
    blocks_per_grid_x = np.ceil(img.shape[0] / threads_per_block[0]).astype(int)
    blocks_per_grid_y = np.ceil(img.shape[1] / threads_per_block[1]).astype(int)
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    autocontrast_kernel[blocks_per_grid, threads_per_block](img_gpu,im_terrarium_gpu)
    cuda.synchronize()

    return im_terrarium_gpu.copy_to_host()

directory = './64thesis/'
directory_target = './64thesis/'
lst = os.listdir(directory)
count = 0
for filename in lst:
    count += 1
    print(count)
    img = cv2.imread(directory + filename,cv2.IMREAD_UNCHANGED)
    im_terrarium = Image.new(mode="RGB", size=(img.shape[0], img.shape[1]))
    print(im_terrarium)
    array = asarray(img)
    terr_array = asarray(im_terrarium)
    
    array.setflags(write=True)
    ImageTerrarium = autocontrast(array,terr_array)
    
    Image.fromarray(ImageTerrarium).save(directory_target+filename)