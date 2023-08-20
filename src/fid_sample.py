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
from torchvision.transforms import functional as F
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image
import argparse
import logging
import os

from model_structures.model_uncondition import DiT_Uncondition_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

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
    with torch.no_grad():
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
            
        ckpt = torch.load(os.path.join(checkpoints_dir, checkpoint), map_location=torch.device(f'cuda:{device}'))
        model = DiT_Uncondition_models[args.model](input_size=latent_size).to(device)
        model.load_state_dict(ckpt['model'])

        model = DDP(model, device_ids=[rank]) 
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
        
        dist.barrier()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_Uncondition_models.keys()), default="DiT_Uncondition-S/4")
    parser.add_argument("--image-size", type=int, choices=[128, 224, 256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fid_samples", type=int, default=6000)
    #parser.add_argument("--example_samples", type=int, default=50)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    args = parser.parse_args()
    main(args)
