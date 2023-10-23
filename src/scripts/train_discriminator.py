# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
import logging
import argparse
from glob import glob
from time import time
from PIL import Image
import torch
import torch.nn as nn
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.datasets import ImageFolder
from diffusers.models import AutoencoderKL

from utils.utils import *
from model_structures.resnet import ResNet
from model_structures.conditional_resnet import ConditionResNet

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger: only log when rang == 0
        logging.basicConfig(
            level=logging.INFO, 
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a ImageNet Classifier using MLPMixer.
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
    input_size = args.image_size // 8

    if args.model == 'resnet':
        model = ResNet(resolution=input_size, num_classes=1000)
    elif args.model == 'condition_resnet':
        model = ConditionResNet(input_size=input_size, class_num=2)
    else:
        raise ValueError(f'{args.model} is not supported.')
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # Setup an experiment folder:
    exp_name = args.experiment_name
    if rank == 0:
        if args.data_path.endswith('/'):
            args.data_path = args.data_path[:-1]
        dataset_name = args.data_path.split('/')[-1]
        model_string_name = args.model 
        experiment_index = len(glob(f"{args.results_dir}/{exp_name}-{model_string_name}-{dataset_name}*"))
        experiment_dir = f"{args.results_dir}/{exp_name}-{model_string_name}-{dataset_name}-{experiment_index:03d}"  
        os.makedirs(experiment_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info("Arguments:")
        for k, v in vars(args).items():
            logger.info(f'{k}: {v}')
    else:
        logger = create_logger(None)
    
    
    train_steps = 0
    # Resume training: continue ckpt args.resume
    if args.resume:
        if not os.path.isfile(args.resume):
            raise ValueError(f'checkpoint dir not exist: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{device}'))

        model.load_state_dict(checkpoint["model"])
        if rank==0:
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        del checkpoint
    else:
        if rank == 0:
            logger.info(f'Build model: {args.model}')
    if args.model == 'resnet': 
        model.resnet.fc = nn.Linear(model.resnet.fc.in_features, 2)
    if rank == 0:
        logger.info(f"{args.model} parameters: {sum(p.numel() for p in model.parameters()):,}")
    # DataParrallel
    model = DDP(model.to(device), device_ids=[rank]) 
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    # Setup data:
    batch_size=int(args.global_batch_size // dist.get_world_size())
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)

    train_sampler, train_loader = get_ddp_sampler_loader(dataset=dataset,
                                num_replicas=dist.get_world_size(),
                                rank=rank,
                                sample_shuffle=True,
                                seed=args.global_seed,
                                batch_size=batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=True)
        
    logger.info(f"Train Dataset contains {len(dataset):,} images")

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    correct = 0
    total = 0
    epoch_correct = 0
    epoch_total = 0
    epoch_loss = 0.0
    
    logger.info(f"Total epoch number: {args.epochs}.")
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        logger.info(f"Begin epoch: {epoch}")
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                
            if args.model == "condition_resnet":
                t = torch.randint(0, 50, (x.shape[0],), device=device)
                x = model(x, t)
            else:
                x = model(x)
            loss = criterion(x, y)
            epoch_loss += loss.item()
            _, pred = torch.max(x.data, -1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            
            epoch_total += y.size(0)
            epoch_correct += (pred == y).sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every_step == 0 and train_steps != 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                logger.info(f"(step={train_steps:08d}) Loss: {avg_loss:.4f}, Running acc: {(correct/total):.3f}, Epoch acc: {(epoch_correct/epoch_total):.3f},  Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                correct = 0
                total = 0
                running_loss = 0
                log_steps = 0
                start_time = time()

        # Save DiT checkpoint:
        if (epoch % args.ckpt_every_epoch == 0 and epoch != 0) or epoch == args.epochs -1:
            
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch":epoch+1,
                    "args": args,
                    "experiment_dir":experiment_dir,
                    "train_steps": train_steps,
                }
                checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path_fin = os.path.join(checkpoint_dir, f'{epoch:08d}.pt')
                torch.save(checkpoint, checkpoint_path_fin)
                logger.info(f"Saved checkpoint to {checkpoint_path_fin}")
            dist.barrier()
        
        logger.info(f"Epoch: {epoch} Accuracy: {(epoch_correct/epoch_total):.4f} Epoch Loss: {(epoch_loss/epoch_total):.4f}")
        epoch_correct = 0
        epoch_total = 0
        epoch_loss = 0
        
    if rank==0:
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet", choices=['mlpmixer', 'resnet', 'condition_resnet'])
    parser.add_argument("--experiment_name", type=str, default="pretrain_discriminator")
    parser.add_argument("--data_path", type=str, required=True, default="datasets/gan_data")
    parser.add_argument("--results_dir", type=str, default="results")
    
    parser.add_argument("--image_size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log_every_step", type=int, default=100)
    parser.add_argument("--ckpt_every_epoch", type=int, default=10)
       
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
    args = parser.parse_args()
    main(args)
