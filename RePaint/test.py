# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import shutil
from PIL import Image
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from chunkgen import *

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402



def shift_right_half(directory,iter):
    # Get a list of all files in the directory
    files = os.listdir(directory)

    for file in files:
        # Check if the file is an image
        if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Open the image
            image_path = os.path.join(directory, file)
            image = Image.open(image_path)

            # Get the dimensions of the image
            width, height = image.size

            # Define the box for the right half of the image
            right_half_box = ((width // 2)-1, 0, width, height)

            # Crop the right half of the image
            right_half = image.crop(right_half_box)

            # Create a new image with the left half empty
            left_half = Image.new('RGBA', (width, height))

            # Paste the right half onto the left half
            left_half.paste(right_half, (0, 0))

            # Save the modified image
            modified_image_path = os.path.join(directory,str(iter)+".png")
            left_half.save(modified_image_path)
            os.remove(image_path)
            

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf,width,img_size):
    
    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))


    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    
    chunks, iterator = chunkgenSetup(width)
    
    for x in range(iterator):
        
        chunks = chunkgeniter(img_size,chunks,x,width,"./chunkgenOutput/",conf_arg["data"]["eval"]["lama_inet256_genhalf_n100_test"]["mask_path"],conf_arg["data"]["eval"]["lama_inet256_genhalf_n100_test"]["gt_path"])
        
        all_images = []

        dset = 'eval'

        eval_name = conf.get_default_eval_name()
        
        
        dl = conf.get_dataloader(dset=dset, dsName=eval_name)

               
        for batch in iter(dl):
            
            for k in batch.keys():
                if isinstance(batch[k], th.Tensor):
                    batch[k] = batch[k].to(device)

            model_kwargs = {}

            model_kwargs["gt"] = batch['GT']
            
            

            gt_keep_mask = batch.get('gt_keep_mask')
            if gt_keep_mask is not None:
                model_kwargs['gt_keep_mask'] = gt_keep_mask

            batch_size = model_kwargs["gt"].shape[0]

            if conf.cond_y is not None:
                classes = th.ones(batch_size, dtype=th.long, device=device)
                model_kwargs["y"] = classes * conf.cond_y
            else:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(batch_size,), device=device
                )
                model_kwargs["y"] = classes

            sample_fn = (
                diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
            )


            result = sample_fn(
                model_fn,
                (batch_size, 3, conf.image_size, conf.image_size),
                clip_denoised=conf.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=show_progress,
                return_all=True,
                conf=conf
            )
            srs = toU8(result['sample'])
            gts = toU8(result['gt'])
            lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                    th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

            gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

            conf.eval_imswrite(
                srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
                img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)
            
        split_images(conf_arg["data"]["eval"]["lama_inet256_genhalf_n100_test"]["paths"]["srs"],"./chunkgenOutput")
        

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument('--width', type=int,required=False,default=4)
    parser.add_argument('--img_size', type=int,required=False,default=64)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    iter_arg = args.get('width')
    size_arg = args.get('img_size')
    main(conf_arg,iter_arg,size_arg)
