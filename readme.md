# Introduction

This Readme will quickly go over how to reproduce all the results from Infinite Adjacent Heightmap Generation
with Image Outpainting.
You might have to do certain neccessary setup steps for the 4 repos used in this method.

https://github.com/openai/guided-diffusion

https://github.com/andreas128/RePaint

https://github.com/victorca25/traiNNer/tree/master

https://github.com/victorca25/iNNfer

If you do not wish to train your own models, download the Models.zip file from here:

https://drive.google.com/file/d/1AlpCfDimwXjHwVMuHpr5LdYFyeHbZldH/view?usp=sharing

and unpack it into the ./models folder


## Obtaining the Dataset

Sadly, i cannot provide the dataset on github since it is about 300gb in size.
It can however be obtained here: https://search.earthdata.nasa.gov/downloads/4436602377
for free. An account will be neccesary however. Once an Account is created, you can use the download.sh script after inserting your own username.

## Preprocessing

After the dataset is obtained, it has to be sliced in 256 x 256 slices. This can be done via slicing.py

to8bit.py autocontrasts the grayscale slices.

to64.py downscales files to 64x64

terranize.py applies the terrarium encoding

Do not forget. In order to train an upscaler on the terrarium data, a downscaled copy should be created first. After that both the 256x and 64x files should be terranized individually.

## Training

To train a diffusion model you simply have to run **bash train.sh** in ./guided-diffusion . You might have to adjust the image size and the directory for training images.

To train an upscaler model you have to run python **train.py -opt options/sr/train_sr.yml** inside of ./traiNNer/codes . You will have to adjust the dataroots for train and val inside of train_sr.yml .

## Sampling

To sample from the trained diffusion model, simply call **bash sample.sh** in ./guided-diffusion Dont forget to adjust image size and model path. 

The resulting samples will be stored in ./guided-diffusion/samples as .npz file

By running **python sampleout.py** in ./guided-diffusion , those samples can be stored as images again in ./guided-diffusion/output

## Outpainting

To outpaint, place all images you want outpainted in a folder, this is set to ./guided-diffusion/output by default. 

Then adjust all neccesary parameters inside ./RePaint/outpaintAll.py like image size, desired output size in chunks and the correct config file. Make sure you pick an even output size.

Once this is done, you can run **python outpaintAll.py** and all images inside the input folder will automatically be extended to the desired chunk size and combined.

## Upscaling

If you have worked with the low res version till now, you can now upscale your results.

To do this simply move them into ./iNNfer/fullImages 

Then inside ./iNNfer ,you call **python all.py** after the upsampling is done, stitch them back together with **python stitch.py**

## Own work

Only some of the scripts are my onw work. Most of the code belongs to the owners of their respective repositories.
These scripts are my own work:
to64.py
to8bit.py
terranize.py
slicing.py
RePaint/outpaintAll.py
RePaint/chunkgen.py
RePaint/combine.py
guided-diffusion/sampleout.py
iNNfer/stitch.py
iNNfer/all.py

