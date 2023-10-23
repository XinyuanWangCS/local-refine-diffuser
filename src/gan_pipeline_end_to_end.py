# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
import time
import math
import pytz
import random
import argparse
from tqdm import tqdm
from copy import deepcopy
from glob import glob
from datetime import datetime

import torch
import torch.nn as nn
import itertools
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.reserved_memory = 0
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


import imageio
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from model_structures.model_uncondition import DiT_Uncondition_models
from model_structures.resnet import ResNet
from model_structures.conditional_resnet import ConditionResNet
from utils.utils import str2bool, requires_grad, create_logger, update_ema, get_sampler_and_loader

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def generate_q_sequence_samples(
    diffusion,
    loader,
    save_dir, 
    rank, 
    device,
    batch_size, 
    start_t, 
    end_t, 
    interval,
    logger = None,
    vae = None,
    num_samples=1280,
    seed=None, 
    **kwargs):
    
    global_batch_size = batch_size * dist.get_world_size()
    total_samples = int(math.ceil(num_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // batch_size)
    
    if rank == 0 and logger is not None:
        logger.info(f"Total number of images that will be sampled: {total_samples}")
        logger.info(f'sample needed :{samples_needed_this_gpu}, iterations: {iterations}')
    
    save_indices = list(range(start_t, end_t, interval))[::-1]
    os.makedirs(save_dir, exist_ok=True)
    for t in save_indices:
        os.makedirs(os.path.join(save_dir, 'real', f'{t:04d}'), exist_ok=True)
        
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
    if rank == 0 and logger is not None:
        logger.info(f"Total number of images that will be sampled: {total_samples}")
        logger.info(f'sample needed :{samples_needed_this_gpu}, iterations: {iterations}')
    
    save_indices = list(range(start_t, end_t, interval))[::-1]
    os.makedirs(save_dir, exist_ok=True)
    for t in save_indices:
        os.makedirs(os.path.join(save_dir, 'fake', f'{t:04d}'), exist_ok=True)
        
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
            
#################################################################################
#                                  Training Loop                                #
#################################################################################

def build_logger(args, rank):
    # Setup an experiment folder:
    exp_name = args.experiment_name
    if rank == 0:
        if args.data_dir.endswith('/'):
            args.data_dir = args.data_dir[:-1]
        dataset_name = args.data_dir.split('/')[-1]
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        pacific = pytz.timezone('America/Los_Angeles')
        pacific_time = datetime.now(pacific)
        formatted_time = pacific_time.strftime("%Y%m%d%H%M")
        experiment_dir = f"{args.results_dir}/{formatted_time}-{exp_name}-{dataset_name}-{model_string_name}" 
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info('------------------------------------------')
        logger.info(f'Build model: {args.model}')
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info("Arguments:")
        for k, v in vars(args).items():
            logger.info(f'{k}: {v}')
        logger.info('------------------------------------------')
    else:
        logger = create_logger(None)
    return experiment_dir, logger

def build_discriminator(args, device, latent_size, rank, logger):
    if args.discriminator == 'resnet': #[batchsize, 2048]
        discriminator = ResNet(resolution=latent_size, num_classes=1000)
        discriminator.resnet.fc = nn.Linear(discriminator.resnet.fc.in_features, 1)
        discriminator = discriminator.to(device)
    elif args.model == 'condition_resnet':
        discriminator = ConditionResNet(input_size=latent_size, class_num=1)
    else:
        raise ValueError(f'{args.discriminator} is not supported.')

    # Load discriminator ckpt
    discriminator_ckpt = None
    if args.discriminator_ckpt:
        if not os.path.isfile(args.discriminator_ckpt):
            raise ValueError(f'discriminator checkpoint dir does not exist: {args.discriminator_ckpt}.')

        discriminator_ckpt = torch.load(args.discriminator_ckpt, map_location=torch.device(f'cpu'))        
        discriminator.load_state_dict(discriminator_ckpt['model'])
        del discriminator_ckpt
        if rank==0:
            logger.info("=> load encoder checkpoint '{}'".format(args.discriminator_ckpt))

    return discriminator

def sample_discriminator_training_data(sample_dir, model, diffusion, vae, logger, rank, seed, device, latent_size, batch_size, args):
    
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
            start_t=args.start_t,
            end_t = args.end_t,
            interval = args.interval,
            num_samples=args.num_samples, 
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
            start_t=args.start_t,
            end_t = args.end_t,
            interval = args.interval,
            num_samples=args.num_samples, 
            data_dir=args.data_dir,
            )
        
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group("nccl") # backend: NVIDIA Collective Communications Library（NCCL）
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    batch_size=int(args.global_batch_size // dist.get_world_size())
    rank = dist.get_rank()
    if args.use_seed:
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)  
    else:
        seed = None
        
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    experiment_dir, logger = build_logger(args, rank)
        
    # Setup data:
    sampler, loader = get_sampler_and_loader(args, args.data_dir, rank, logger)
    data_iter = itertools.cycle(loader)
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule
    
    model = DiT_Uncondition_models[args.model](
        input_size=latent_size
    )
    ema = deepcopy(model).to(device)
    # Resume training: continue ckpt args.resume
    if args.resume:
        if not os.path.isfile(args.resume):
            raise ValueError(f'checkpoint dir not exist: {args.resume}')
        if rank ==0: print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cpu'))

        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        del checkpoint
    
    # DataParrallel
    model = DDP(model.to(device), device_ids=[rank]) 
    requires_grad(ema, False)
    ema.eval()
    
    # Build discriminator
    discriminator = build_discriminator(args, device, latent_size, rank, logger)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    diffusion_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    disciminator_optimizer= torch.optim.AdamW(discriminator.parameters(), lr=1e-4, weight_decay=0)
    gan_loss_func = nn.BCELoss()
    sample_indices = list(range(args.start_t, args.end_t, args.interval))
    
    # Variables for monitoring/logging purposes:
    iteration = 10
    for iter in range(iteration):
        logger.info(f"-----------------------------------------")
        logger.info(f"Beginning iteration {iter}...")
        
        # sample real and fake data
        logger.info(f"Tteration {iter}: Sampling...")
        sample_dir = os.path.join(args.save_dir, f'iter{iter}')
        sample_discriminator_training_data(sample_dir, model, diffusion, vae, logger, rank, seed, device, latent_size, batch_size, args)
        
        # Train Discriminator
        logger.info(f"-----------------------------------------")
        logger.info(f"Tteration {iter}: Train Discriminator...")
        running_loss = 0.0
        iter_correct = 0
        iter_total = 0
        iter_loss = 0.0
        
        discriminator.train()
        requires_grad(discriminator, True)
        real_loader = get_sampler_and_loader(args, os.path.join(sample_dir, "real"), rank, logger)
        fake_loader = get_sampler_and_loader(args, os.path.join(sample_dir, "fake"), rank, logger)
        real_data_iter = itertools.cycle(real_loader)
        fake_data_iter = itertools.cycle(fake_loader)
        logger.info(f"Total step number: {args.total_steps}.")
        start_time = time.time()
        for step in range(args.dis_total_steps+1):
            real, real_t = next(real_data_iter)
            fake, fake_t = next(fake_data_iter)
            if step % len(real_loader) == 0 and step != 0:
                sampler.set_epoch(step // len(real_loader))
                logger.info(f'Epoch: {step // len(real_loader)} Step: {step}')
            with torch.no_grad():
                real = vae.encode(real.to(device)).latent_dist.sample().mul_(0.18215)
                fake = vae.encode(fake.to(device)).latent_dist.sample().mul_(0.18215)
            real_preds = discriminator(real, real_t)
            fake_preds = discriminator(fake, fake_t)
            real_loss = gan_loss_func(real_preds, torch.ones(real_preds.shape[0], device=device))
            fake_loss = gan_loss_func(fake_preds, torch.zeros(fake_preds.shape[0], device=device))
            loss = real_loss + fake_loss
            disciminator_optimizer.zero_grad()
            loss.backward()
            disciminator_optimizer.step()

            running_loss += loss.item()
            iter_loss += loss.item()
            total = real_preds.shape[0] + fake_preds.shape[0]
            iter_total += real_preds.shape[0] + fake_preds.shape[0]
            real_count = ((real_preds >= 0.5) == 1).sum().item()
            fake_count = ((fake_preds < 0.5) == 0).sum().item()
            correct = real_count + fake_count
            iter_correct += real_count + fake_count
            
            if step % args.log_every_step == 0:
                log_steps += args.log_every_step
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                correct = torch.tensor(correct, device=device)
                dist.all_reduce(correct, op=dist.ReduceOp.SUM)
                correct = correct.item()
                total = torch.tensor(total, device=device)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                total = total.item()
                
                if rank == 0:
                    logger.info(f"(step={step:07d}) Discriminator Loss: {avg_loss:.4f} GAN acc={(correct/total):.3f} ({correct}/{total}), Train Steps/Sec: {steps_per_sec:.2f} ")
                
                # Reset monitoring variables:
                running_loss = 0
                correct = 0
                total = 0
                start_time = time.time()
        
        avg_loss = torch.tensor(iter_loss / (args.dis_total_steps+1), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()

        correct = torch.tensor(iter_correct, device=device)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        correct = correct.item()
        total = torch.tensor(iter_total, device=device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        total = total.item()
        
        if rank == 0:
            checkpoint = {"model": discriminator.module.state_dict()}
            checkpoint_dir = os.path.join(experiment_dir, 'discriminator_checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path_fin = os.path.join(checkpoint_dir, f'{step:06d}.pt')
            torch.save(checkpoint, checkpoint_path_fin)
            logger.info(f"Saved discriminator checkpoint to {checkpoint_path_fin}")
            
            logger.info(f"(Iteration={iter}) Discriminator Loss: {avg_loss:.4f} GAN acc={(correct/total):.3f} ({correct}/{total}), Train Steps/Sec: {steps_per_sec:.2f} ")

        log_steps = 0
        dif_running_loss = 0
        gan_running_loss = 0
        running_loss = 0
        
        correct = 0
        total = 0
        iter_correct = 0
        iter_total = 0
        iter_loss = 0.0
        iter_dif_loss = 0.0
        iter_gan_loss = 0.0
        
        model.train()
        discriminator.eval()
        requires_grad(discriminator, False)
        logger.info(f"Total step number: {args.total_steps}.")
        
        start_time = time.time()
        for step in range(args.total_steps+1):
            x, _ = next(data_iter)
            x = x.to(device)
            
            if step % len(loader) == 0 and step != 0:
                sampler.set_epoch(step // len(loader))
                logger.info(f'Epoch: {step // len(loader)} Step: {step}')
            
            labels = torch.ones((x.size(0)), dtype=torch.int64, requires_grad = False).to(device)
            # train diffusion model 
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            t = random.sample(sample_indices, (x.shape[0],))
            t = torch.tensor(t, device=device)
            
            loss_dict = diffusion.training_losses_output_xt(model, x, t)
            pred_x = loss_dict["pred"]
            
            dis_preds = discriminator(pred_x, t)#extract_resnet_perceptual_outputs_v0(encoder, pred)
            gan_loss = gan_loss_func(dis_preds, labels)
            
            pred_labels = dis_preds >= 0.5
            total += labels.size(0)
            correct += (pred_labels == labels).sum().item()
            iter_total += labels.size(0)
            iter_correct += (pred_labels == labels).sum().item()
            
            t = torch.randint(0, args.num_sampling_steps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t)
            dif_loss = loss_dict["loss"].mean()
            
            loss = dif_loss + args.alpha * gan_loss
            diffusion_optimizer.zero_grad()
            loss.backward()
            diffusion_optimizer.step()
            
            update_ema(ema, model.module)
            # Log loss values:
            dif_running_loss += dif_loss.item()
            gan_running_loss += gan_loss.item()
            running_loss += loss.item()

            iter_loss += loss.item()
            iter_dif_loss += dif_loss.item()
            iter_gan_loss += gan_loss.item()
            
            if step % args.log_every_step == 0:
                log_steps += args.log_every_step
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                dif_avg_loss = torch.tensor(dif_running_loss / log_steps, device=device)
                dist.all_reduce(dif_avg_loss, op=dist.ReduceOp.SUM)
                dif_avg_loss = dif_avg_loss.item() / dist.get_world_size()

                gan_avg_loss = torch.tensor(gan_running_loss / log_steps, device=device)
                dist.all_reduce(gan_avg_loss, op=dist.ReduceOp.SUM)
                gan_avg_loss = gan_avg_loss.item() / dist.get_world_size()

                correct = torch.tensor(correct, device=device)
                dist.all_reduce(correct, op=dist.ReduceOp.SUM)
                correct = correct.item()
                total = torch.tensor(total, device=device)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                total = total.item()
                
                if rank == 0:
                    logger.info(f"(step={step:07d}) Loss: {avg_loss:.4f} Dif_Loss: {dif_avg_loss:.4f}, GAN_Loss: {args.alpha * gan_avg_loss:.4f}, GAN acc={(correct/total):.3f} ({correct}/{total}), Train Steps/Sec: {steps_per_sec:.2f} ")
                
                # Reset monitoring variables:
                running_loss = 0
                dif_running_loss = 0
                gan_running_loss = 0
                correct = 0
                total = 0
                log_steps = 0
                start_time = time.time()

            # Save DiT checkpoint:
            if step % args.ckpt_every_step == 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "d_opt": diffusion_optimizer.state_dict(),
                        "args": args,
                        "experiment_dir":experiment_dir,
                        "train_steps": step,
                        "ema": ema.state_dict(),
                    }
                    
                    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path_fin = os.path.join(checkpoint_dir, f'{step:08d}.pt')
                    torch.save(checkpoint, checkpoint_path_fin)
                    logger.info(f"Saved checkpoint to {checkpoint_path_fin}")
            
        iter_correct = torch.tensor(iter_correct, device=device)
        dist.all_reduce(iter_correct, op=dist.ReduceOp.SUM)
        iter_correct = iter_correct.item()
        iter_total = torch.tensor(iter_total, device=device)
        dist.all_reduce(iter_total, op=dist.ReduceOp.SUM)
        iter_total = iter_total.item()
        
        logger.info(f"(Iteration {iter}) step={step:07d} Loss: {iter_loss/iter_steps:.4f} Dif_Loss: {iter_dif_loss/iter_steps:.4f}, GAN_Loss: {args.alpha * iter_gan_loss/step:.4f}, GAN acc={(iter_correct/iter_total):.3f} ({iter_correct}/{iter_total})")
        iter_correct = 0
        iter_total = 0
        iter_loss = 0.0
        iter_dif_loss = 0.0
        iter_gan_loss = 0.0
        iter_steps = 0
        dist.barrier()
        
            
if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="gan_pipeline_end_to_end")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_Uncondition_models.keys()), default="DiT_Uncondition-B/4")
    parser.add_argument("--image_size", type=int, choices=[128, 224, 256, 512], default=256)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--dis_total_steps", type=int, default=1000)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every_step", type=int, default=20)
    parser.add_argument("--ckpt_every_step", type=int, default=10000)
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    # synthesis data
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--start_t", type=int, default=0)
    parser.add_argument("--end_t", type=int, default=50)
    parser.add_argument("--interval", type=int, default=100)
    
    parser.add_argument('--use_seed', type=str2bool, default=True)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--discriminator", type=str, default="condition_resnet", choices=['mlpmixer', 'condition_resnet', 'resnet'])
    parser.add_argument(
        "--discriminator_ckpt",
        default="",
        type=str,
        metavar="PATH",
        help="path to the discriminator's checkpoint",
    )
    args = parser.parse_args()
    main(args)

    
    
    




    

