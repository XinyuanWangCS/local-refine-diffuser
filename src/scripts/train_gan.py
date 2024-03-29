# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""


import math
from tqdm import tqdm
import torch
import torch.nn as nn
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from model_uncondition import *

from transformers import CLIPVisionModel

#from datasets import load_dataset

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
#                                  Discriminator                                #
#################################################################################

#class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, input):
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)

        attn_weights = nn.functional.softmax(q @ k.transpose(-2, -1), dim=-1)
        output = attn_weights @ v

        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.features = 4 * 32 * 32

        self.model = nn.Sequential(
            nn.Linear(self.features, 4096),
            nn.GroupNorm(1, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(4096, 2048),
            nn.GroupNorm(1, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(2048, 1024),
            nn.GroupNorm(1, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(1024, 512),
            nn.GroupNorm(1, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 256),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1) 
        validity = self.model(img_flat)
        return validity

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

    BCELoss = torch.nn.BCELoss()

    # Setup an experiment folder:
    if rank == 0:
        experiment_index = len(glob(f"{args.results_dir}/*"))
        dataset_name = args.data_path.split('/')[-1]
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{dataset_name}-{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(experiment_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        args_logger = logging.getLogger("args logger")
        args_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(experiment_dir, 'args.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        args_logger.addHandler(file_handler)
        arg_dict = vars(args)
        for key, value in arg_dict.items():
            args_logger.info(f"{key}: {value}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model = DiT_Uncondition_models[args.model]( 
        input_size=latent_size
    )
    print(f'Build model: {args.model}')

    # Note that parameter initialization is done within the DiT constructor
    #ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    #requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank]) # DataParrallel
    discriminator = Discriminator().to(device)
    
    diffusion = create_diffusion(str(num_sampling_steps))  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt_g = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, weight_decay = 0)
    
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
        seed=args.global_seed # 似乎应该是rank specific seed
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

    # Prepare models for training:
    #update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    #ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0 # the actual loss backward in generator
    gen_running_loss = 0 # generator loss from DiT
    g_adv_running_loss = 0 # adversarial loss
    dis_running_loss = 0 # sum loss
    start_time = time()


    tau = args.tau

    if rank != 0:
        experiment_index = len(glob(f"{args.results_dir}/*"))-1
        dataset_name = args.data_path.split('/')[-1]
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/encoder_loss-{dataset_name}-{experiment_index:03d}-{model_string_name}"  # Create an experiment folder

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, _ in loader:
            x = x.to(device)

            #print(f'x shape: {x.shape}')

            gt = torch.ones((x.size(0), 1), requires_grad = False).to(device)
            fake = torch.zeros((x.size(0), 1), requires_grad = False).to(device)


            #print(x)

            # -----------------
            #  train diffusion model 
            # -----------------

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict() #y=y
            loss_dict = diffusion.training_losses_step_output(model, x, t, model_kwargs)
            pred_xt, gt_xt = loss_dict["pred_xt"], loss_dict["gt_xt"]
            dm_loss = loss_dict["loss"].mean()

            #print(f'pred_xt shape: {pred_xt.shape}')

            preds = discriminator(pred_xt)
            #print(f'preds shape: {preds.shape}')
            g_adv_loss = BCELoss(preds, gt)

            #print(f'diffusion loss: {dm_loss}')
            #print(f'adversarial loss: {g_adv_loss}')

            # 0.8 tau (0.2 for g_adv_loss)
            loss = tau * dm_loss + (1-tau) * g_adv_loss
            opt_g.zero_grad()
            loss.backward()
            opt_g.step()
            #update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            gen_running_loss += dm_loss.item()
            g_adv_running_loss += g_adv_loss.item()
            

            # -----------------
            #  train discriminator model 
            # -----------------

            real_loss = BCELoss(discriminator(gt_xt), gt)
            fake_loss = BCELoss(discriminator(pred_xt.detach()), fake)

            discriminator_loss = (real_loss + fake_loss) / 2
            

            opt_d.zero_grad()
            discriminator_loss.backward()
            opt_d.step()
            
            dis_running_loss += discriminator_loss.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                gen_avg_loss = torch.tensor(gen_running_loss / log_steps, device=device)
                dist.all_reduce(gen_avg_loss, op=dist.ReduceOp.SUM)
                gen_avg_loss = gen_avg_loss.item() / dist.get_world_size()

                g_adv_avg_loss = torch.tensor(g_adv_running_loss / log_steps, device=device)
                dist.all_reduce(g_adv_avg_loss, op=dist.ReduceOp.SUM)
                g_adv_avg_loss = g_adv_avg_loss.item() / dist.get_world_size()

                dis_avg_loss = torch.tensor(dis_running_loss / log_steps, device=device)
                dist.all_reduce(dis_avg_loss, op=dist.ReduceOp.SUM)
                dis_avg_loss = dis_avg_loss.item() / dist.get_world_size()

                logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f} Generator_Loss: {gen_avg_loss:.4f}, Generator_Adversarial_Loss: {g_adv_avg_loss:.4f}, Discriminator Loss: {dis_avg_loss: .4f} Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                gen_running_loss = 0
                g_adv_running_loss = 0
                dis_running_loss = 0

                log_steps = 0
                start_time = time()

        # Save DiT checkpoint:
        if epoch % args.ckpt_every == 0 or epoch == args.epochs -1:
            if rank == 0:
                checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save DiT model
                checkpoint_dit = {
                    "model": model.module.state_dict(),
                    #"ema": ema.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "args": args
                }
                checkpoint_path_fin = os.path.join(checkpoint_dir, f'{epoch:07d}_dit.pt')
                torch.save(checkpoint_dit, checkpoint_path_fin)
                logger.info(f"Saved DiT checkpoint to {checkpoint_path_fin}")
                
                # Save discriminator model
                checkpoint_discriminator = {
                    "model": discriminator.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "args": args
                }
                checkpoint_path_fin = os.path.join(checkpoint_dir, f'{epoch:07d}_discriminator.pt')
                torch.save(checkpoint_discriminator, checkpoint_path_fin)
                logger.info(f"Saved discriminator checkpoint to {checkpoint_path_fin}")

            dist.barrier()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_Uncondition_models.keys()), default="DiT_Uncondition-B/4")
    parser.add_argument("--image-size", type=int, choices=[128, 224, 256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--ckpt_every", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--fid_samples", type=int, default=1000)
    parser.add_argument("--example_samples", type=int, default=50)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    args = parser.parse_args()
    main(args)
