from matplotlib import pyplot as plt
from IPython import display
from PIL import Image
import numpy as np
import math
from patchify import patchify, unpatchify
import os
from datetime import datetime


startTime = datetime.now()
# Input
directory = './AsterBatch/'
# Output
savepath = './Batch256/'

lst = os.listdir(directory)
filecount = 0
print("Number of Files: " + str(len(lst)))
for filename in lst:
    try:
        img = Image.open(directory + filename)
    
        img_array = np.asarray(img)
        #print(img_array.shape)
        patches = patchify(img_array,(256,256),step=256)
        count = 0
        filecount += 1
        print("CurrentFile: " + str(filecount))
        #print(patches.shape)
        for row in patches:
            for patch in row:
                Image.fromarray(patch).save(savepath + os.path.splitext(filename)[0] + str(count) + ".png")
                count += 1
    except Exception as e:
        print(f"Error processing image {filename}: {e}")
        continue  # move on to the next image
    


