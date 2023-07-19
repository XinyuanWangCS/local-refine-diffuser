# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import math
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image  # for saving generated samples
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from torchvision import transforms
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from cleanfid import fid
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from model_uncondition import DiT_Uncondition_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters()) # Dict(name, parameter)
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay) # add_用于权重缩减, ema_params = decay * ema_params + (1-decay) * parram
        # param.data绕过自动微分，不会计算param的梯度 => param.detach()

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger: 只有排名为0的进程执行logging
        logging.basicConfig(
            level=logging.INFO, # 记录级别为INFO
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                          Sample images for FID                                #
#################################################################################

def generate_samples(ckpt_str, fid_dir, model, diffuser, vae, rank, device, latent_size=32, num_samples=1000, n=16):
    # Create random noise for input
    global_batch_size = n * dist.get_world_size()
    
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(num_samples / global_batch_size) * global_batch_size)

    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f'sample needed :{samples_needed_this_gpu}, iterations: {iterations}')

    ckpt_fid_samples_dir = os.path.join(fid_dir, ckpt_str)
    os.makedirs(ckpt_fid_samples_dir, exist_ok=True)

    total = 0
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    
    for _ in pbar:
        z = torch.randn((n, 4, latent_size, latent_size)).to(device)
        samples = diffuser.p_sample_loop(
            model.forward, z.shape, z, clip_denoised = False, device = device
        )
        
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, img in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            if index >= num_samples:
                return
            Image.fromarray(img).save(f'{ckpt_fid_samples_dir}/{index:07d}.png')
            
        total += global_batch_size


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl") # backend: NVIDIA Collective Communications Library（NCCL）
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    num_sampling_steps = args.num_sampling_steps
    batch_size=int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    experiment_dir = args.experiment_dir
    if not os.path.exists(experiment_dir):
        raise ValueError(f'Experiment dir not exist: {experiment_dir}')

    fid_samples_dir = os.path.join(experiment_dir, 'fid_samples')
    os.makedirs(fid_samples_dir, exist_ok=True)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    if rank == 0:
        print(f'Build model: {args.model}')
    diffusion = create_diffusion(str(num_sampling_steps))  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        raise ValueError(f'checkpoints dir not exist: {checkpoints_dir}')
    checkpoints = sorted(os.listdir(checkpoints_dir))

    for checkpoint in checkpoints:
        if rank == 0:
            print('----------------------------------------------')
            print(f'Sampling: {checkpoint}')
        model = DiT_Uncondition_models[args.model](input_size=latent_size)

        ckpt = torch.load(os.path.join(checkpoints_dir, checkpoint))
        model.load_state_dict(ckpt['model'])

        model = DDP(model.to(device), device_ids=[rank]) # DataParrallel
        model.eval()

        ckpt_name = checkpoint.split('.')[0]
        with torch.no_grad():
            generate_samples(ckpt_str = ckpt_name,
                                fid_dir = fid_samples_dir,
                                model=model, 
                                diffuser=diffusion, 
                                vae=vae, 
                                rank=rank, 
                                device=device, 
                                latent_size=latent_size,
                                num_samples=args.fid_samples, 
                                n=batch_size)
        if rank == 0:    
            print(f"Saved {args.fid_samples} images for {ckpt_name}th epoch")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_Uncondition_models.keys()), default="DiT_Uncondition-S/4")
    parser.add_argument("--image-size", type=int, choices=[128, 224, 256, 512], default=224)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fid_samples", type=int, default=6000)
    #parser.add_argument("--example_samples", type=int, default=50)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    args = parser.parse_args()
    main(args)
