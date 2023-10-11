# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
import time
import argparse
import logging
from copy import deepcopy
from PIL import Image
from glob import glob
from itertools import count
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.reserved_memory = 0
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from model_structures.model_uncondition import DiT_Uncondition_models
from model_structures.resnet import ResNet


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
   
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
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO, 
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

    # Create model:
    train_steps = 0
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    model = DiT_Uncondition_models[args.model](
        input_size=latent_size
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule
    
    # Resume training: continue ckpt args.resume
    if args.resume:
        if not os.path.isfile(args.resume):
            raise ValueError(f'checkpoint dir not exist: {args.resume}')
        if rank ==0: print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{device}'))

        if args.load_ema:
            model.load_state_dict(checkpoint["ema"])
            if rank ==0: print(f"=> loaded ema checkpoint {args.resume}")   
        else:
            model.load_state_dict(checkpoint["model"])
            if rank ==0: print(f"=> loaded non-ema checkpoint {args.resume}")
        
        ema.load_state_dict(checkpoint["ema"])
        del checkpoint
    
    # DataParrallel
    model = DDP(model.to(device), device_ids=[rank]) 
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    dif_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    # Setup an experiment folder:
    exp_name = args.experiment_name
    if rank == 0:
        if args.data_path.endswith('/'):
            args.data_path = args.data_path[:-1]
        dataset_name = args.data_path.split('/')[-1]
        experiment_index = len(glob(f"{args.results_dir}/{exp_name}-{dataset_name}*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{exp_name}-{dataset_name}-{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info('------------------------------------------')
        logger.info(f'Build model: {args.model}')
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info("Arguments:")
        for k, v in vars(args).items():
            logger.info(f'{k}: {v}')
        logger.info('------------------------------------------')
    else:
        logger = create_logger(None)
    
    discriminator_ckpt = None
    if args.discriminator_ckpt:
        if not os.path.isfile(args.discriminator_ckpt):
            raise ValueError(f'discriminator checkpoint dir does not exist: {args.discriminator_ckpt}.')

        discriminator_ckpt = torch.load(args.discriminator_ckpt, map_location=torch.device(f'cuda:{device}'))        
        
        if rank==0:
            logger.info("=> load encoder checkpoint '{}'".format(args.discriminator_ckpt))
        
    if args.discriminator == 'resnet': #[batchsize, 2048]
        discriminator = ResNet(resolution=latent_size, num_classes=1000)
        discriminator.resnet.fc = nn.Linear(discriminator.resnet.fc.in_features, 2)
        discriminator = discriminator.to(device)
        if args.discriminator_ckpt:
            discriminator.load_state_dict(discriminator_ckpt['model'])
            del discriminator_ckpt
            if rank==0:
                logger.info(f"=> load discriminator state_dict")
    else:
        raise ValueError(f'{args.discriminator} is not supported.')

    requires_grad(discriminator, False)
    discriminator.eval()
    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed 
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
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    epoch_steps = 0
    dif_running_loss = 0
    gan_running_loss = 0
    running_loss = 0
    
    gan_loss_func = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    epoch_correct = 0
    epoch_total = 0
    epoch_loss = 0.0
    epoch_dif_loss = 0.0
    epoch_gan_loss = 0.0
    
    
    logger.info(f"Total step number: {args.total_steps}.")
    start_time = time.time()
    for epoch in count(args.start_epoch):
        model.train()
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, _ in loader:
            x = x.to(device)
            labels = torch.ones((x.size(0)), dtype=torch.int64, requires_grad = False).to(device)
            # train diffusion model 
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                
            t = torch.randint(0, args.start_step, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses_end_to_end_step(model, x, t)
            pred_x = loss_dict["pred"]
            dif_loss = loss_dict["loss"].mean()
            with torch.no_grad():
                dis_preds = discriminator(pred_x)#extract_resnet_perceptual_outputs_v0(encoder, pred)

            gan_loss = gan_loss_func(dis_preds, labels)
            _, pred_labels = torch.max(dis_preds.data, -1)
            total += labels.size(0)
            correct += (pred_labels == labels).sum().item()
            epoch_total += labels.size(0)
            epoch_correct += (pred_labels == labels).sum().item()
            
            loss = dif_loss + args.alpha * gan_loss
            dif_opt.zero_grad()
            loss.backward()
            dif_opt.step()
            
            update_ema(ema, model.module)
            # Log loss values:
            dif_running_loss += dif_loss.item()
            gan_running_loss += gan_loss.item()
            running_loss += loss.item()

            epoch_loss += loss.item()
            epoch_dif_loss += dif_loss.item()
            epoch_gan_loss += gan_loss.item()
            
            epoch_steps += 1
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every_step == 0:
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
                    logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f} Dif_Loss: {dif_avg_loss:.4f}, GAN_Loss: {args.alpha * gan_avg_loss:.4f}, GAN acc={(correct/total):.3f} ({correct}/{total}), Train Steps/Sec: {steps_per_sec:.2f} ")
                
                # Reset monitoring variables:
                running_loss = 0
                dif_running_loss = 0
                gan_running_loss = 0
                correct = 0
                total = 0
                log_steps = 0
                start_time = time.time()

            # Save DiT checkpoint:
            
            if train_steps % args.ckpt_every_step == 0 or train_steps == args.total_steps -1 or train_steps==1:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "d_opt": dif_opt.state_dict(),
                        "epoch":epoch+1,
                        "args": args,
                        "experiment_dir":experiment_dir,
                        "train_steps": train_steps,
                        "ema": ema.state_dict(),
                    }
                    
                    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path_fin = os.path.join(checkpoint_dir, f'{train_steps:08d}.pt')
                    torch.save(checkpoint, checkpoint_path_fin)
                    logger.info(f"Saved checkpoint to {checkpoint_path_fin}")
            
            if train_steps >= args.total_steps:
                logger.info("Done!")
                return
        
        epoch_correct = torch.tensor(epoch_correct, device=device)
        dist.all_reduce(epoch_correct, op=dist.ReduceOp.SUM)
        epoch_correct = epoch_correct.item()
        epoch_total = torch.tensor(epoch_total, device=device)
        dist.all_reduce(epoch_total, op=dist.ReduceOp.SUM)
        epoch_total = epoch_total.item()
        
        logger.info(f"(Epoch {epoch}) step={train_steps:07d} Loss: {epoch_loss/epoch_steps:.4f} Dif_Loss: {epoch_dif_loss/epoch_steps:.4f}, GAN_Loss: {args.alpha * epoch_gan_loss/epoch_steps:.4f}, GAN acc={(epoch_correct/epoch_total):.3f} ({epoch_correct}/{epoch_total})")
        epoch_correct = 0
        epoch_total = 0
        epoch_loss = 0.0
        epoch_dif_loss = 0.0
        epoch_gan_loss = 0.0
        epoch_steps = 0
        dist.barrier()

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="gan_samplet")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_Uncondition_models.keys()), default="DiT_Uncondition-B/4")
    parser.add_argument("--image_size", type=int, choices=[128, 224, 256, 512], default=256)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every_step", type=int, default=20)
    parser.add_argument("--ckpt_every_step", type=int, default=10000)
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--start_step", type=int, default=50)
    parser.add_argument('--load_ema', type=str2bool, default=False)
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
    parser.add_argument("--discriminator", type=str, default="resnet", choices=['mlpmixer', 'biggan', 'resnet'])
    parser.add_argument(
        "--discriminator_ckpt",
        default="",
        type=str,
        metavar="PATH",
        help="path to the discriminator's checkpoint",
    )
    args = parser.parse_args()
    main(args)

    
    
    




    

