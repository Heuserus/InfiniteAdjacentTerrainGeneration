import os
from PIL import Image
import random

def combine_images(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') and not f.startswith('-')]

    if not image_files:
        print("No image files found in the folder.")
        return

    # Determine the maximum x and y coordinates
    max_x, max_y = 0, 0
    for file_name in image_files:
        coordinates = file_name[:-4].split('_')
        x, y = int(coordinates[0]), int(coordinates[1])
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    # Calculate the size of the combined image
    image_size = (max_x + 1) * 32, (max_y + 1) * 32  # Assuming each image is 100x100 pixels

    # Create a blank canvas for the combined image
    combined_image = Image.new('RGB', image_size)

    # Paste each image onto the combined image based on its coordinates
    for file_name in image_files:
        coordinates = file_name[:-4].split('_')
        x, y = int(coordinates[0]), int(coordinates[1])
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path)
        combined_image.paste(image, (x * 32, y * 32))

    # Save the combined image
    combined_image.save('combined_image'+ str(random.randint(0,100000)) + '.png')
    print("Images combined successfully!")

folder_path = './chunkgenOutput'  
combine_images(folder_path)