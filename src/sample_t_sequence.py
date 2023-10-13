# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
import math
import argparse
from tqdm import tqdm

import torch
from torch import nn
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.transforms import functional as F
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


from model_structures.model_uncondition import DiT_Uncondition_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import imageio

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
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

def generate_sequence_samples(
    model, diffusion, vae, 
    save_dir, 
    rank, device, 
    latent_size, batch_size, 
    start_t=0, end_t=50, internal=1,
    num_samples=1280,
    logger=None, 
    seed=None, 
    **kwargs):
    
    global_batch_size = batch_size * dist.get_world_size()
    total_samples = int(math.ceil(num_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // batch_size)
    if rank == 0:
        if logger is not None:
            logger.info(f"Total number of images that will be sampled: {total_samples}")
            logger.info(f'sample needed :{samples_needed_this_gpu}, iterations: {iterations}')
        else:
            print(f"Total number of images that will be sampled: {total_samples}")
            print(f'sample needed :{samples_needed_this_gpu}, iterations: {iterations}')
    
    save_indices = list(range(start_t, end_t, internal))[::-1]
    os.makedirs(save_dir, exist_ok=True)
    for i in save_indices:
        os.makedirs(os.path.join(save_dir, 'fake', f'{i:04d}'), exist_ok=True)
        
    total = 0
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    seed_count = 0
    with torch.no_grad():
        for _ in pbar:
            if seed is not None:
                torch.manual_seed(seed + rank + seed_count * dist.get_world_size())
                seed_count += 1
            samples = torch.randn((batch_size, 4, latent_size, latent_size)).to(device)
            indices = list(range(start_t, diffusion.num_timesteps))[::-1]
            for t in indices:
                samples = diffusion.p_sample_loop_progressive_step(
                    model = model.forward, 
                    shape = samples.shape, 
                    t = t, 
                    samples=samples, 
                    clip_denoised = False, 
                    device = device
                )
            
                if t in save_indices:
                    images = vae.decode(samples / 0.18215).sample
                    images = images.permute(0, 2, 3, 1).to("cpu").numpy()

                    for i, img in enumerate(images):
                        index = i + rank * batch_size + total
                        if index >= num_samples:
                            return
                        img = (img + 1) / 2
                        imageio.imwrite(os.path.join(save_dir, 'fake', f'{i:04d}', '{index:05d}.tiff'), img)
                    
            total += global_batch_size


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
    seed = args.global_seed #TODO
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    num_sampling_steps = args.num_sampling_steps
    batch_size=int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f'Experiment dir not exist: {checkpoint_dir}')

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    if rank == 0:
        print(f'Build model: {args.model}')
    diffusion = create_diffusion(str(num_sampling_steps))  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    ckpt_name = checkpoint_dir.split('/')[-3]+"-"+checkpoint_dir.split('/')[-1].split('.')[0]
    if rank == 0:
        print('----------------------------------------------')
        print(f'Sampling: {ckpt_name}')
    model = None
    ckpt = torch.load(checkpoint_dir, map_location=torch.device(f'cuda:{device}'))
    model = DiT_Uncondition_models[args.model](input_size=latent_size).to(device)
    
    if args.use_ema:
        model.load_state_dict(ckpt["ema"])
        print("=> loaded ema checkpoint (epoch {})".format(
                ckpt["epoch"]))
    else:
        model.load_state_dict(ckpt["model"])
        print("=> loaded checkpoint (epoch {})".format(
                    ckpt["epoch"]))
    
    del ckpt
    ckpt = None
    model = DDP(model, device_ids=[rank]) 
    requires_grad(model=model, flag=False)
    model.eval()

    with torch.no_grad():
        generate_sequence_samples(
            model=model,
            diffusion=diffusion,
            vae=vae, 
            logger=None,
            ckpt_name = ckpt_name,
            save_dir = save_dir,
             
            
            rank=rank, 
            device=device, 
            latent_size=latent_size,
            start_t=args.start_t,
            end_t = args.end_t,
            num_samples=args.fid_samples, 
            n=batch_size,
            seed=seed)
    if rank == 0:    
        print(f"Saved {args.fid_samples} images for {ckpt_name}th epoch")
    del model
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default='results/samples')
    parser.add_argument("--model", type=str, choices=list(DiT_Uncondition_models.keys()), default="DiT_Uncondition-B/4")
    parser.add_argument("--image-size", type=int, choices=[128, 224, 256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fid_samples", type=int, default=30000)
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--end_t", type=int, default=50)
    parser.add_argument("--start_t", type=int, default=50)
    parser.add_argument('--use_ema', type=str2bool, default=False)
    parser.add_argument('--use_seed', type=str2bool, default=False)
    args = parser.parse_args()
    main(args)
