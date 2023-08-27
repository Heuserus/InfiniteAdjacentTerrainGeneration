import numpy as np
from PIL import Image

# Input the path of the current npz file
data = np.load('./samples/samples_16x64x64x3.npz')
lst = data.files
# Set the size of the samples to either 64 or 256
size = 64
# Set output folder
output = "./output/"

# Initialize a list to store PIL images
pil_images = []

# Convert numpy arrays to PIL images
for item in lst:
    numpy_array = data[item]
    for index,image_array in enumerate(numpy_array):
        Image.fromarray(image_array).save(output+"sample" + str(index) +".png")
        



