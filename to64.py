from PIL import Image
import os

def resize_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Check if the file is an image
        if file.endswith(('.png', '.jpg', '.jpeg','.tif')):
            # Open the image
            image_path = os.path.join(input_folder, file)
            image = Image.open(image_path)

            # Resize the image to 64x64 pixels
            resized_image = image.resize((64, 64))

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, file)
            resized_image.save(output_path)

            print(f"Resized {file} successfully.")

# Specify the input and output folders
input_folder = 'FolderThesis'
output_folder = '22'

# Resize the images
resize_images(input_folder, output_folder)