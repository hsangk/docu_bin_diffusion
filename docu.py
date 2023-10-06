import argparse, os, sys, glob
import PIL
import albumentations
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from skimage.util.shape import view_as_windows

def patchify(img, patch_shape, overlap):

    # Pad image to ensure dimensions are divisible by patch shape
    pad_row = patch_shape[0] - img.shape[0] % patch_shape[0]
    pad_col = patch_shape[1] - img.shape[1] % patch_shape[1]
    img = np.pad(img, [(0, pad_row), (0, pad_col)], mode='constant')

    # Extract patches with overlap
    patches = view_as_windows(img, patch_shape, step=patch_shape[0]-overlap)
    # Reshape patches
    patches = patches.reshape(-1, *patch_shape)
    return patches

def unpatchify(patches, image_shape, overlap):

    # Determine output shape
    rows = (image_shape[0] + patch_shape[0] - 1) // patch_shape[0]
    cols = (image_shape[1] + patch_shape[1] - 1) // patch_shape[1]
    # Reshape patches into original image shape
    image = np.zeros((rows*patch_shape[0], cols*patch_shape[1]), dtype=patches.dtype)
    count = np.zeros((rows*patch_shape[0], cols*patch_shape[1]), dtype=patches.dtype)

    for r in range(rows):
        for c in range(cols):
            patch = patches[r * cols + c]
            r_offset = r * (patch_shape[0] - overlap)
            c_offset = c * (patch_shape[1] - overlap)
            image[r_offset:r_offset+patch_shape[0], c_offset:c_offset+patch_shape[1]] += patch
            count[r_offset:r_offset+patch_shape[0], c_offset:c_offset+patch_shape[1]] += 1
    # Calculate average for overlapping regions
    image = image.astype('float')

    image /= count
    image = image.astype(('uint8'))

    # Remove padding
    image = image[:image_shape[0], :image_shape[1]]
    return image

def make_batch(image, device):
    rescaler = albumentations.SmallestMaxSize(max_size=256)
    cropper = albumentations.CenterCrop(256, 256)
    preprocessor = albumentations.Compose([rescaler, cropper])

    image = np.array(image).astype(np.uint8)
    image = preprocessor(image=image)["image"]
    image = np.expand_dims((image / 127.5 - 1.0).astype(np.float32), axis=2)

    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    batch = {"image": image}

    for k in batch:
        batch[k] = batch[k].to(device=device)
    return batch



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        default='./inference/images',
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        default='./inference/outputs',
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default='./weights/diffusion/best.ckpt',
        help="location of your best weight"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./inference/ldm-4-3-1.yaml",
        help="location of inference config"
    )

    opt = parser.parse_args()

    images = sorted(glob.glob(os.path.join(opt.indir, "*.*")))
    print(f"Found {len(images)} inputs.")

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.weights)["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for im in tqdm(images):
                step = 0
                step += 1
                outpath = os.path.join(opt.outdir, os.path.split(im)[1])
                image = np.array(Image.open(im).convert("L"))

                patch_shape = (256, 256)
                overlap = 0
                patches = patchify(image, patch_shape, overlap)

                image_result = np.zeros((patches.shape), dtype=image.dtype)
                for i in range(patches.shape[0]):
                    print(f'Image : {im}, Patches : {i} / {patches.shape[0]}')
                    img = patches[i]

                    batch = make_batch(img, device=device)
                    x = torch.randn((1, 3, 64, 64), device=device)
                    c = model.cond_stage_model.encode(batch["image"])

                    x = torch.cat((x, c), dim=1)
                    shape = (x.shape[1]-3,)+x.shape[2:]
                    samples_ddim, _ = sampler.sample(S=opt.steps,
                                                     conditioning=c,
                                                     batch_size=x.shape[0],
                                                     shape=shape,
                                                     verbose=False)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                  min=0.0, max=1.0)
                    patch_result = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255

                    image_result[i] = patch_result[:,:,0]

                image_reconstructed = unpatchify(image_result, image.shape, overlap)
                Image.fromarray(image_reconstructed.squeeze().astype(np.uint8)).save(outpath)