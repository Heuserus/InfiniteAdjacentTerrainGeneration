import os
import subprocess
import shutil
from PIL import Image

def getstartfile(img_size,input_image,output_folder,desired_size):
    chunk_mid = (desired_size // 2)-1
    img = Image.open(input_image)
    half = img_size // 2
    top_left = img.crop((0, 0, half, half)).save(f"{output_folder}/{str(chunk_mid)}_{str(chunk_mid)}.png")
    top_right = img.crop((half, 0, img_size, half)).save(f"{output_folder}/{str(chunk_mid+1)}_{str(chunk_mid)}.png")
    bottom_left = img.crop((0, half, half, img_size)).save(f"{output_folder}/{str(chunk_mid)}_{str(chunk_mid+1)}.png")
    bottom_right = img.crop((half, half, img_size, img_size)).save(f"{output_folder}/{str(chunk_mid+1)}_{str(chunk_mid+1)}.png")
    

def main():
    input_folder = "../guided-diffusion/output"
    output_folder = "chunkgenOutput"
    BASH_SCRIPT_PATH = "sample.sh"
    desired_size = 6
    img_size = 64
    conf_path = "confs/64x64grayscale.yml"
    

    
    for input_file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, input_file_name)

        print(f"Cleaning up files in {output_folder}...")
        for filename in os.listdir(output_folder):
            if not filename.startswith("-"):
                file_path = os.path.join(output_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

        getstartfile(img_size,input_file_path,output_folder,desired_size)
        
        print(f"Calling main python script")
        process = subprocess.run(["python","test.py","--conf_path",str(conf_path),"--width",str(desired_size),"--img_size",str(img_size)])

        print(f"Calling combine.py")
        subprocess.run(["python", "combine.py"])
                     

if __name__ == "__main__":
    main()