import os
import cv2

# Define the input and output directories
input_dir = './Val256'
output_dir = './Val256'

count = 0
for filename in os.listdir(input_dir):
    count +=1
    if filename.endswith('.png'):
        # Load the 16-bit grayscale image
        img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_ANYDEPTH)

        # Convert the image to 8-bit grayscale
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Save the converted image to the output directory
        output_filename = filename
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        print(count) 