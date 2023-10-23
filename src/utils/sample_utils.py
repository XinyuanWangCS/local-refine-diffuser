import os
import math
import torch
import imageio
import torch.distributed as dist
from tqdm import tqdm
from .utils import get_sampler_and_loader

def generate_q_sequence_samples(
    diffusion,
    save_dir, 
    rank, 
    device,
    batch_size, 
    args,
    logger = None,
    vae = None,
    num_samples=1280,
    seed=None, 
    ):
    
    global_batch_size = batch_size * dist.get_world_size()
    total_samples = int(math.ceil(num_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // batch_size)
    
    if rank == 0 and logger is not None:
        logger.info(f"Total number of images that will be sampled: {total_samples}")
        logger.info(f'sample needed :{samples_needed_this_gpu}, iterations: {iterations}')
    
    save_indices = list(range(args.start_t, args.end_t, args.interval))[::-1]
    os.makedirs(save_dir, exist_ok=True)
    for t in save_indices:
        os.makedirs(os.path.join(save_dir, 'real', f'{t:04d}'), exist_ok=True)
    
    sampler, loader = get_sampler_and_loader(args, args.data_dir, rank, logger)
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
                images = diffusion.q_sample(x, t_tensor)
                if vae is not None:
                    images = vae.decode(images / 0.18215).sample
                    images = images.permute(0, 2, 3, 1).to("cpu").numpy()
                for i, img in enumerate(images):
                    index = i + rank * batch_size + total
                    if index >= num_samples:
                        return
                    img = (img + 1) / 2
                    imageio.imwrite(os.path.join(save_dir, 'real', f'{t:04d}', f'{index:05d}.tiff'), img)
                
            total += global_batch_size


def generate_p_sequence_samples(
    model, 
    diffusion, 
    vae, 
    save_dir, 
    rank, 
    device, 
    latent_size, 
    batch_size, 
    args,
    num_samples=1280,
    logger=None, 
    seed=None, 
    ):
    
    global_batch_size = batch_size * dist.get_world_size()
    total_samples = int(math.ceil(num_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // batch_size)
    if rank == 0 and logger is not None:
        logger.info(f"Total number of images that will be sampled: {total_samples}")
        logger.info(f'sample needed :{samples_needed_this_gpu}, iterations: {iterations}')
    
    save_indices = list(range(args.start_t, args.end_t, args.interval))[::-1]
    os.makedirs(save_dir, exist_ok=True)
    for t in save_indices:
        os.makedirs(os.path.join(save_dir, 'fake', f'{t:04d}'), exist_ok=True)
        
    total = 0
    pbar = range(iterations)
    pbar = tqdm(pbar, leave=False) if rank == 0 else pbar
    seed_count = 0
    with torch.no_grad():
        for _ in pbar:
            if seed is not None:
                torch.manual_seed(seed + rank + seed_count * dist.get_world_size())
                seed_count += 1
            samples = torch.randn((batch_size, 4, latent_size, latent_size)).to(device)
            indices = list(range(0, diffusion.num_timesteps))[::-1]
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

def sample_discriminator_training_data(
    sample_dir, model, diffusion, vae, logger, rank, seed, device, latent_size, batch_size, args
    ):  
    with torch.no_grad():
        model.eval()
        generate_p_sequence_samples(
            model=model,
            diffusion=diffusion,
            vae=vae, 
            logger=logger,
            save_dir = sample_dir,
            rank=rank, 
            seed=seed,
            device=device, 
            latent_size=latent_size,
            batch_size=batch_size,
            num_samples=args.num_samples, 
            args=args
            )

        generate_q_sequence_samples(
            vae=vae,
            device=device,
            diffusion=diffusion,
            seed=seed,
            logger=logger,
            save_dir = sample_dir, 
            rank=rank, 
            batch_size=batch_size,
            num_samples=args.num_samples, 
            args=args
            )