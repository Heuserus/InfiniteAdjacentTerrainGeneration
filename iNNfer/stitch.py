from PIL import Image
import os
import math

def stitch_subfolder(subfolder_path):
    side = round(math.sqrt(len(os.listdir(subfolder_path))))
    size = side*256
    output_image = Image.new('RGB', (size, size))

    for i in range(side):
        for j in range(side):
            chunk_path = os.path.join(subfolder_path, f'chunk_{(j*64)}_{(i*64)}.png')
            chunk_image = Image.open(chunk_path)
            x_offset = j * 256
            y_offset = i * 256
            output_image.paste(chunk_image, (x_offset, y_offset))

    return output_image

def main():
    folder_path = './256outputs'
    output_folder = './finaloutputs'


    # Iterate through all subfolders in the main folder
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)

        if os.path.isdir(subfolder_path):
            # Stitch the subfolder's images into a single image
            output_image = stitch_subfolder(subfolder_path)

            # Save the output image
            output_image_path = os.path.join(output_folder, f'{subfolder_name}_stitched.png')
            output_image.save(output_image_path)

if __name__ == '__main__':
    print("Start Stitching")
    main()