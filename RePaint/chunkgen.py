import math
from PIL import Image
import os
import cv2

def chunkgenSetup(width):
    iterValue = (width - 2) / 2
    iter = round(iterValue * iterValue + iterValue)
    
    rows, cols = (width, width)
    chunks = [[0 for i in range(cols)] for j in range(rows)]
    
    chunks[round(width/2-1)][round(width/2-1)] = 1
    chunks[round(width/2)][round(width/2-1)] = 1
    chunks[round(width/2-1)][round(width/2)] = 1
    chunks[round(width/2)][round(width/2)] = 1
    
    return chunks, iter
    

    
def chunkgeniter(img_size,chunks,i,width,chunkdir,maskdir,gendir):
    
    clear(maskdir)
    clear(gendir)
    
    if i < (width-2)/2:
        top = [[[round(width/2-1),round(width/2-2-i)],[round(width/2),round(width/2-2-i)]],[[round(width/2-1),round(width/2-1-i)],[round(width/2),round(width/2-1-i)]]]
        bot = [[[round(width/2-1),round(width/2+i)],[round(width/2),round(width/2+i)]],[[round(width/2-1),round(width/2+1+i)],[round(width/2),round(width/2+1+i)]]]
        left = [[[round(width/2-2-i),round(width/2-1)],[round(width/2-1-i),round(width/2-1)]],[[round(width/2-2-i),round(width/2)],[round(width/2-1-i),round(width/2)]]]
        right = [[[round(width/2+i),round(width/2-1)],[round(width/2+1+i),round(width/2-1)]],[[round(width/2+i),round(width/2)],[round(width/2+1+i),round(width/2)]]]
        iterBlocks = [top,bot,left,right]
    else:
        i2 = i % ((width-2)/2)
        i3 = math.floor(i / ((width-2)/2)) -1
        print(i / ((width-2)/2))
        
        tr = [[[round(width/2+i3),round(width/2-2-i2)],[round(width/2+1+i3),round(width/2-2-i2)]],[[round(width/2+i3),round(width/2-1-i2)],[round(width/2+1+i3),round(width/2-1-i2)]]]
        br = [[[round(width/2+i3),round(width/2+i2)],[round(width/2+1+i3),round(width/2+i2)]],[[round(width/2+i3),round(width/2+1+i2)],[round(width/2+1+i3),round(width/2+1+i2)]]]
        bl = [[[round(width/2-2-i3),round(width/2+i2)],[round(width/2-1-i3),round(width/2+i2)]],[[round(width/2-2-i3),round(width/2+1+i2)],[round(width/2-1-i3),round(width/2+1+i2)]]]
        tl = [[[round(width/2-2-i3),round(width/2-2-i2)],[round(width/2-1-i3),round(width/2-2-i2)]],[[round(width/2-2-i3),round(width/2-1-i2)],[round(width/2-1-i3),round(width/2-1-i2)]]]
        iterBlocks = [tr,br,bl,tl]
        
    
    constructImageAndMask(img_size,chunks,iterBlocks,chunkdir,maskdir,gendir)
    for block in iterBlocks:
        for row in block:
            for chunk in row:
                chunks[chunk[0]][chunk[1]] = 1
                
    return chunks
    


def constructImageAndMask(img_size,chunks,iterBlocks,chunkdir,maskdir,gendir):
    
    blackchunk = Image.new('RGB', (img_size // 2,img_size // 2), color=(0, 0, 0))
    whitechunk = Image.new('RGB', (img_size // 2,img_size // 2), color=(255, 255, 255))
    
    
    for p, iterBlock in enumerate(iterBlocks):
        smallchunks = []
        maskchunks = []
        iterBlockName = ""
        for row in iterBlock:
            for coordinates in row:
                x,y = coordinates
                if chunks[x][y] == 1:
                    
                    filename = f"{x}_{y}.png"
                    smallchunk = Image.open(chunkdir+filename)
                    smallchunks.append(smallchunk)
                    maskchunks.append(whitechunk)
                else:
                    smallchunks.append(blackchunk)
                    maskchunks.append(blackchunk)
                iterBlockName += f"_{x}_{y}"
    
        iterBlockImage = Image.new('RGB', (img_size,img_size), color=(255, 255, 255))
        mask = Image.new('RGB', (img_size,img_size), color=(255, 255, 255))

        for i, image in enumerate(smallchunks):
            row = i // 2
            col = i % 2
            x = col * img_size // 2
            y = row * img_size // 2
            iterBlockImage.paste(image,(x,y))
        
        for i,image in enumerate(maskchunks):
            row = i // 2
            col = i % 2
            x = col * img_size // 2
            y = row * img_size // 2
            mask.paste(image,(x,y))
           
        iterBlockImage.save(gendir + "/" + iterBlockName+".png")
        mask.save(maskdir + "/" + iterBlockName+"mask.png")


def split_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    file_list = os.listdir(input_folder)

    for file_name in file_list:
        if file_name.endswith('.png'):
            
            input_path = os.path.join(input_folder, file_name)
            image = cv2.imread(input_path)

           
            top_left = image[:img_size // 2, :img_size // 2]
            top_right = image[:img_size // 2, img_size // 2:img_size]
            bottom_left = image[img_size // 2:img_size, :img_size // 2]
            bottom_right = image[img_size // 2:img_size, img_size // 2:img_size]

           
            coords = file_name.split('.')[0].split('_')  
            cv2.imwrite(os.path.join(output_folder, coords[1] + "_" + coords[2] + '.png'), top_left)
            cv2.imwrite(os.path.join(output_folder, coords[3] + "_" + coords[4] + '.png'), top_right)
            cv2.imwrite(os.path.join(output_folder, coords[5] + "_" + coords[6] + '.png'), bottom_left)
            cv2.imwrite(os.path.join(output_folder, coords[7] + "_" + coords[8] + '.png'), bottom_right)

    print("Image splitting completed.")
    clear(input_folder)




def clear(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)