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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

from torchvision.transforms import functional as F
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


from model_structures.model_uncondition import DiT_Uncondition_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


def plot_images_from_dir(directory, image_num, subfolder_name='fake'):
    if not os.path.exists(os.path.join(directory, subfolder_name)):
        print(os.path.join(directory, subfolder_name), )
        return
    
    save_path = os.path.join(directory, 'examples', subfolder_name)
    os.makedirs(save_path, exist_ok=True)
    
    folder_names = os.listdir(os.path.join(directory, subfolder_name))
    print(folder_names)
    for folder in folder_names:
        image_files = [os.path.join(directory,subfolder_name,folder, f) for f in os.listdir(os.path.join(directory,subfolder_name,folder)) if f.endswith('.tiff')]
        image_files = sorted(image_files)[:image_num]
        
        rows = (image_num + 3) // 4
        fig, axs = plt.subplots(rows, 4, figsize=(5*4, 5*rows))
        
        for idx, ax in enumerate(axs.ravel()):
            if idx < len(image_files):
                img = imageio.imread(image_files[idx])
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')
        
        # Set title to the name of the directory with a larger font size
        fig.suptitle(os.path.basename(directory), fontsize=25)
        
        # Ensure no spacing between subplots and adjust space at the top
        plt.subplots_adjust(wspace=0, hspace=0, top=0.97)
        plt.savefig(os.path.join(save_path, folder), bbox_inches='tight', pad_inches=0)
        plt.close()
    
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
def generate_q_sequence_samples(
    diffusion,
    data_dir,
    save_dir, 
    rank, 
    device,
    batch_size, 
    start_t, 
    end_t, 
    interval,
    vae = None,
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
            
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(data_dir, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    iterations = int(num_samples // batch_size)
    if rank == 0:
        if logger is not None:
            logger.info(f"Total number of images that will be sampled: {num_samples}")
            logger.info(f'iterations: {iterations}')
        else:
            print(f"Total number of images that will be sampled: {num_samples}")
            print(f'iterations: {iterations}')
    
    save_indices = list(range(start_t, end_t, interval))[::-1]
    os.makedirs(save_dir, exist_ok=True)
    for i in save_indices:
        os.makedirs(os.path.join(save_dir, 'real', f'{i:04d}'), exist_ok=True)
        print(f'make dir: {save_dir}/real/{i:04d}')
        
    total = 0
    pbar = enumerate(loader)
    pbar = tqdm(pbar) if rank == 0 else pbar
    seed_count = 0
    with torch.no_grad():
        for i, (x, y) in pbar:
            if seed is not None:
                torch.manual_seed(seed + rank + seed_count * dist.get_world_size())
                seed_count += 1
            if vae is not None:
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = x.to(device)
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                
            for t in save_indices:
                t_tensor = torch.tensor([t] * batch_size, device=device)
                samples = diffusion.q_sample(x, t_tensor)
                if vae is not None:
                    images = vae.decode(samples / 0.18215).sample
                    images = images.permute(0, 2, 3, 1).to("cpu").numpy()
                for i, img in enumerate(images):
                    index = i + rank * batch_size + total
                    if index >= num_samples:
                        return
                    img = (img + 1) / 2
                    imageio.imwrite(os.path.join(save_dir, 'real', f'{t:04d}', f'{index:05d}.tiff'), img)
                
            total += global_batch_size


def generate_p_sequence_samples(
    model, diffusion, vae, 
    save_dir, 
    rank, 
    device, 
    latent_size, 
    batch_size, 
    start_t, 
    end_t, 
    interval,
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
    
    save_indices = list(range(start_t, end_t, interval))[::-1]
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
                        imageio.imwrite(os.path.join(save_dir, 'fake', f'{t:04d}', f'{index:05d}.tiff'), img)
                    
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
    if args.use_seed:
        seed = args.global_seed #TODO
        torch.manual_seed(seed)
    else:
        seed = None
    
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    num_sampling_steps = args.num_sampling_steps
    batch_size=int(args.global_batch_size // dist.get_world_size())
    
    # Setup an experiment folder:
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f'Experiment dir not exist: {checkpoint_dir}')

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    if rank == 0:
        print(f'Build model: {args.model}')
    diffusion = create_diffusion(str(num_sampling_steps))  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    if args.sample_p:
        ckpt_name = checkpoint_dir.split('/')[-3]+"-"+checkpoint_dir.split('/')[-1].split('.')[0]
        if rank == 0:
            print('----------------------------------------------')
            print(f'Sampling: {ckpt_name}')
        model = None
        ckpt = torch.load(checkpoint_dir, map_location=torch.device(f'cuda:{device}'))
        model = DiT_Uncondition_models[args.model](input_size=latent_size).to(device)
        
        if args.load_ema:
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
            generate_p_sequence_samples(
                model=model,
                diffusion=diffusion,
                vae=vae, 
                logger=None,
                save_dir = args.save_dir,
                rank=rank, 
                device=device, 
                latent_size=latent_size,
                batch_size=batch_size,
                start_t=args.start_t,
                end_t = args.end_t,
                interval = args.interval,
                num_samples=args.num_samples, 
                seed=seed)
    
    if args.sample_q:
        generate_q_sequence_samples(
            vae=vae,
            device=device,
            diffusion=diffusion,
            data_dir=args.data_dir,
            save_dir = args.save_dir, 
            rank=rank, 
            batch_size = batch_size, 
            start_t=args.start_t,
            end_t = args.end_t,
            interval = args.interval,
            num_samples=args.num_samples, 
            seed=seed)
        
    if args.plot_examples:
        plot_images_from_dir(directory=args.save_dir, 
                             image_num=8, 
                             subfolder_name='fake')
        plot_images_from_dir(directory=args.save_dir, 
                             image_num=8, 
                             subfolder_name='real')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default='datasets/celebahq256')
    parser.add_argument("--save_dir", type=str, default='results/sample_t_sequence')
    parser.add_argument("--model", type=str, choices=list(DiT_Uncondition_models.keys()), default="DiT_Uncondition-B/4")
    parser.add_argument("--image-size", type=int, choices=[128, 224, 256, 512], default=256)
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--start_t", type=int, default=0)
    parser.add_argument("--end_t", type=int, default=50)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument('--load_ema', type=str2bool, default=False)
    parser.add_argument('--use_seed', type=str2bool, default=True)
    parser.add_argument('--plot_examples', type=str2bool, default=False)
    parser.add_argument('--sample_p', type=str2bool, default=True)
    parser.add_argument('--sample_q', type=str2bool, default=True)
    args = parser.parse_args()
    main(args)
