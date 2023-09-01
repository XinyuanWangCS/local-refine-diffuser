# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from glob import glob
from time import time
import argparse

from model_structures.mlp_mixer import MLPMixerClassifier
from model_structures.biggan_classifier import BigGANClassifier
from diffusers.models import AutoencoderKL
from datasets import load_dataset
from collections import OrderedDict
from copy import deepcopy
import logging
from utils.utils import *

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
    if args.model == 'biggan':
        model = BigGANClassifier(in_channels=4, resolution=input_size, output_dim=1000)
    elif args.model == 'mlpmixer':
        model = MLPMixerClassifier(in_channels=4, image_size=input_size, patch_size=4, num_classes=1000,
                 dim=768, depth=12, token_dim=196, channel_dim=1024)
    if args.use_ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        ema.eval()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    train_steps = 0
    # Resume training: continue ckpt args.resume
    if args.resume:
        if not os.path.isfile(args.resume):
            raise ValueError(f'checkpoint dir not exist: {args.resume}')
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{device}'))
        args.start_epoch = checkpoint["epoch"]
        if args.use_ema:
            model.load_state_dict(checkpoint["ema"])
            print("=> loaded ema checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]))
        else:
            model.load_state_dict(checkpoint["model"])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]))
        experiment_dir = checkpoint["experiment_dir"]
        train_steps = checkpoint["train_steps"]
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
    else:
        if rank == 0:
            print(f'Build model: {args.model}')

    # DataParrallel
    model = DDP(model.to(device), device_ids=[rank]) 
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0)
    
    criterion = nn.CrossEntropyLoss()
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        checkpoint = None
        
    # Setup an experiment folder:
    exp_name = args.experiment_name
    if not args.resume:
        if rank == 0:
            dataset_name = args.data_path.split('/')[-1]
            experiment_index = len(glob(f"{args.results_dir}/{exp_name}-{dataset_name}*"))
            model_string_name = args.model  
            experiment_dir = f"{args.results_dir}/{exp_name}-{dataset_name}-{experiment_index:03d}--{model_string_name}"  # Create an experiment folder
            os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            os.makedirs(experiment_dir, exist_ok=True)

            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")
            logger.info("Arguments:")
            for k, v in vars(args).items():
                logger.info(f'{k}: {v}')
        else:
            logger = create_logger(None)

    logger.info(f"{args.model} parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup data:
    dataset = load_dataset("imagenet-1k",cache_dir=args.data_path)
    transform = get_transform(image_size=args.image_size)
    train_dataset = ImageDataset(dataset['train'], transform)
    #validation_dataset = ImageDataset(dataset['validation'], transform)
    test_dataset = ImageDataset(dataset['validation'], transform)

    batch_size=int(args.global_batch_size // dist.get_world_size())
    train_sampler, train_loader = get_ddp_sampler_loader(dataset=train_dataset,
                                num_replicas=dist.get_world_size(),
                                rank=rank,
                                sample_shuffle=True,
                                seed=args.global_seed,
                                batch_size=batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=True)
    if rank == 0:
        test_sampler, test_loader = get_ddp_sampler_loader(dataset=test_dataset, # 可能改成无DDP
                                    num_replicas=1,
                                    rank=rank,
                                    sample_shuffle=False,
                                    seed=args.global_seed,
                                    batch_size=batch_size,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    drop_last=True)
        
        logger.info(f"Train Dataset contains {len(train_dataset):,} images")
        #logger.info(f"Validation Dataset contains {len(validation_dataset):,} images")
        logger.info(f"Test Dataset contains {len(test_dataset):,} images")

    if rank != 0 and not args.resume:
        experiment_index = len(glob(f"{args.results_dir}/{exp_name}*"))
        dataset_name = args.data_path.split('/')[-1]
        model_string_name = args.model
        experiment_dir = f"{args.results_dir}/{exp_name}-{dataset_name}-{experiment_index:03d}-{model_string_name}"

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
            
            if args.use_ema:
                update_ema(ema, model.module)
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0 and train_steps != 0:
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
        if (epoch % args.ckpt_every == 0 and epoch != 0) or epoch == args.epochs -1:
            
            if rank == 0:
                if args.use_ema:
                    checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch":epoch+1,
                    "args": args,
                    "experiment_dir":experiment_dir,
                    "train_steps": train_steps,
                    "ema": ema.state_dict(),
                }
                else:
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
        
        if rank == 0 and ((epoch % args.test_every_epoch == 0) or epoch == args.epochs -1):
            torch.cuda.synchronize()
            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0
            with torch.no_grad():
                for x, y in tqdm(test_loader):
                    x, y = x.to(device), y.to(device)
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    x = model(x)
                    loss = criterion(x, y)
                    
                    _, pred = torch.max(x.data, -1)
                    test_total += y.size(0)
                    
                    test_correct += (pred == y).sum().item()
                    test_loss += loss.item()
            logger.info(f"Test epoch: {epoch}  Test Accuracy: {(test_correct/test_total):.4f} Test Loss: {(test_loss/test_total):.4f}")
            
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="biggan", choices=['mlpmixer', 'biggan'])
    parser.add_argument("--experiment-name", type=str, default="imagenet_classifer")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1)
    parser.add_argument("--test-every-epoch", type=int, default=1) 
    parser.add_argument('--use_ema', type=str2bool, default=True)
       
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    args = parser.parse_args()
    main(args)
