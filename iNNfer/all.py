import os
import subprocess
from PIL import Image
import shutil

def split_image(image_path, output_folder):
    img = Image.open(image_path)
    width, height = img.size
    chunk_size = 64

    for i in range(0, width, chunk_size):
        for j in range(0, height, chunk_size):
            box = (i, j, i + chunk_size, j + chunk_size)
            chunk = img.crop(box)
            chunk_filename = os.path.join(output_folder, f"chunk_{i}_{j}.png")
            chunk.save(chunk_filename)

def process_images(input_folder):
    model_path = "GrayscaleUpscale.pth"

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            output_folder = "input"
            os.makedirs(output_folder, exist_ok=True)
            split_image(image_path, output_folder)

            subprocess.run(["python", "run.py", "-m", model_path])
            
            for file in output_folder:
                file_path = os.path.join(output_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            os.mkdir("./256outputs/" + os.path.splitext(filename)[0])
                    
            for file in os.listdir("output"):
                file_path = os.path.join("output", file)
                if os.path.isfile(file_path):
                    
                    destination_path = "./256outputs/" + os.path.splitext(filename)[0]
                    shutil.move(file_path, destination_path)


if __name__ == "__main__":
    input_folder = "fullImages"
    process_images(input_folder)