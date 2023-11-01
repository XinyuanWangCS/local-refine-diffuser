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
from diffusion import create_diffusion
from tqdm import tqdm
from PIL import Image
import argparse
import os
from utils.unet_utils import create_unet_model_and_diffusion, model_and_diffusion_defaults

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

def generate_samples(ckpt_name, save_dir, model, diffusion, rank, device, seed, end_step, latent_size, n, num_samples=1000):
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

    ckpt_fid_samples_dir = os.path.join(save_dir, ckpt_name)
    os.makedirs(ckpt_fid_samples_dir, exist_ok=True)

    total = 0
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    seed_count = 0
    with torch.no_grad():
        for _ in pbar:
            torch.manual_seed(seed + rank + seed_count * dist.get_world_size())
            seed_count += 1
            z = torch.randn((n, 3, latent_size, latent_size), device=device)
            samples = diffusion.p_sample_loop(
                model.forward, z.shape, end_step=end_step, noise=z, clip_denoised = False, device = device,
                progress = True
            )
            
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i, img in enumerate(samples):
                index = i + rank * n + total
                if index >= num_samples:
                    return
                Image.fromarray(img).save(f'{ckpt_fid_samples_dir}/{index:07d}.png')
                
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

    batch_size=int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f'Experiment dir not exist: {checkpoint_dir}')

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model_diffusion_defaults = model_and_diffusion_defaults()
    cifar_args = dict(
        image_size = 32, 
        num_channels = 128,
        num_res_blocks = 3,
        learn_sigma = True,
        dropout = 0.3,
        diffusion_steps = 4000,
        noise_schedual = "cosine",
        )
    model_diffusion_defaults.update(cifar_args)
    model, diffusion = create_unet_model_and_diffusion(**model_diffusion_defaults)
    ckpt = torch.load(checkpoint_dir, map_location=torch.device(f'cpu'))
    model.to(device)

    model.load_state_dict(ckpt)
    del ckpt
    if rank== 0:
        print("=> loaded checkpoint: epoch {}".format(checkpoint_dir))

    model = DDP(model, device_ids=[rank]) 
    requires_grad(model=model, flag=False)
    model.eval()

    with torch.no_grad():
        generate_samples(
            ckpt_name = checkpoint_dir.split('/')[-1],
            save_dir = save_dir,
            model=model, 
            diffusion=diffusion, 
            rank=rank, 
            device=device, 
            latent_size=args.image_size,
            end_step = args.end_step,
            num_samples=args.sample_num, 
            n=batch_size,
            seed=seed)
    if rank == 0:    
        print(f"Saved {args.sample_num} images to {args.save_dir}")        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default='results/samples')
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample_num", type=int, default=128)
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--end_step", type=int, default=0)
    args = parser.parse_args()
    main(args)
