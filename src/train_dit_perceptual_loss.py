# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import torch.nn as nn
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
import numpy as np
from collections import OrderedDict
from PIL import Image
from glob import glob
import time
import argparse
import logging
import os

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from model_structures.model_uncondition import DiT_Uncondition_models

from model_structures.mlp_mixer import MLPMixerClassifier
from model_structures.biggan_classifier import BigGANClassifier
from model_structures.resnet import ResNet
from copy import deepcopy

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
    if args.use_ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        ema.eval()
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # Setup an experiment folder:
    exp_name = args.experiment_name
    if rank == 0:
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
        
    # Resume training: continue ckpt args.resume
    if args.resume:
        if not os.path.isfile(args.resume):
            raise ValueError(f'checkpoint dir not exist: {args.resume}')
        
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{device}'))

        if args.use_ema:
            model.load_state_dict(checkpoint["ema"])
            logger.info(f"=> loaded ema checkpoint {args.resume}")
        else:
            model.load_state_dict(checkpoint["model"])
            logger.info(f"=> loaded non-ema checkpoint {args.resume}")
        del checkpoint
        
    encoder_ckpt = None
    if args.encoder_ckpt:
        if not os.path.isfile(args.encoder_ckpt):
            raise ValueError(f'Encoder checkpoint dir does not exist: {args.encoder_ckpt}.')
    
        encoder_ckpt = torch.load(args.encoder_ckpt, map_location=torch.device(f'cuda:{device}'))
        if rank==0:
            logger.info("=> load encoder checkpoint '{}'".format(args.encoder_ckpt))
    
    if args.perceptual_encoder == 'biggan':
        encoder = BigGANClassifier(in_channels=4, resolution=latent_size, output_dim=1000).to(device)
        if args.encoder_ckpt:
            encoder.load_state_dict(encoder_ckpt['model'])
            if rank==0:
                logger.info("=> load encoder state_dict")
        encoder = encoder.encoder
    elif args.perceptual_encoder == 'mlpmixer':
        encoder = MLPMixerClassifier(in_channels=4, image_size=latent_size, patch_size=4, num_classes=1000,
                 dim=768, depth=12, token_dim=196, channel_dim=1024).to(device)
        if args.encoder_ckpt:
            encoder.load_state_dict(encoder_ckpt['model'])
            if rank==0:
                logger.info("=> load encoder state_dict")
        encoder = encoder.mlp_mixer
    elif args.perceptual_encoder == 'resnet': #[batchsize, 2048]
        encoder = ResNet(resolution=latent_size, num_classes=1000).to(device)
        if args.encoder_ckpt:
            encoder.load_state_dict(encoder_ckpt['model'])
            if rank==0:
                logger.info("=> load encoder state_dict")
        encoder.resnet.avgpool = nn.Identity()
        encoder.resnet.fc = nn.Identity()
    else:
        raise ValueError(f'{args.encoder} is not supported.')
    del encoder_ckpt
    requires_grad(encoder, False)
    encoder.eval()
    
    model = DDP(model.to(device), device_ids=[rank]) # DataParrallel
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    d_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        
    diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule

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

    tau = args.tau
    # Variables for monitoring/logging purposes:
    log_steps = 0
    d_running_loss = 0
    p_running_loss = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Total step number: {args.total_steps}.")
    for epoch in range(args.start_epoch, 100000):
        model.train()
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, _ in loader:
            x = x.to(device)
            # train diffusion model 
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses_step_output(model, x, t)
            pred_xt, gt_xt = loss_dict["pred_xt"], loss_dict["gt_xt"]
            dm_loss = loss_dict["loss"].mean()
            with torch.no_grad():
                feature_pred_xt = encoder(pred_xt)
                feature_gt_xt = encoder(gt_xt)
            
            percept_loss = ((feature_pred_xt - feature_gt_xt)**2).mean()
            
            loss = tau * dm_loss + (1-tau) * percept_loss
            d_opt.zero_grad()
            loss.backward()
            d_opt.step()
            if args.use_ema:
                update_ema(ema, model.module)
            
            # Log loss values:
            p_running_loss += percept_loss.item()
            d_running_loss += dm_loss.item()
            running_loss += loss.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                d_avg_loss = torch.tensor(d_running_loss / log_steps, device=device)
                dist.all_reduce(d_avg_loss, op=dist.ReduceOp.SUM)
                d_avg_loss = d_avg_loss.item() / dist.get_world_size()

                p_avg_loss = torch.tensor(p_running_loss / log_steps, device=device)
                dist.all_reduce(p_avg_loss, op=dist.ReduceOp.SUM)
                p_avg_loss = p_avg_loss.item() / dist.get_world_size()

                logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f} D_Loss: {d_avg_loss:.4f}, P_Loss: {p_avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                d_running_loss = 0
                p_running_loss = 0

                log_steps = 0
                start_time = time.time()

            # Save DiT checkpoint:
            
            if train_steps % args.ckpt_every_step == 0 or train_steps == args.total_steps -1 or train_steps==1:
                if rank == 0:
                    if args.use_ema:
                        checkpoint = {
                        "model": model.module.state_dict(),
                        "d_opt": d_opt.state_dict(),
                        "epoch":epoch+1,
                        "args": args,
                        "experiment_dir":experiment_dir,
                        "train_steps": train_steps,
                        "ema": ema.state_dict(),
                    }
                    else:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "d_opt": d_opt.state_dict(),
                            "epoch":epoch+1,
                            "args": args,
                            "experiment_dir":experiment_dir,
                            "train_steps": train_steps,
                        }
                    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path_fin = os.path.join(checkpoint_dir, f'{train_steps:08d}.pt')
                    torch.save(checkpoint, checkpoint_path_fin)
                    logger.info(f"Saved checkpoint to {checkpoint_path_fin}")
            
            if train_steps >= args.total_steps:
                logger.info("Done!")
                cleanup()
            
        dist.barrier()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="perceptual")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_Uncondition_models.keys()), default="DiT_Uncondition-B/4")
    parser.add_argument("--image-size", type=int, choices=[128, 224, 256, 512], default=256)
    parser.add_argument("--total_steps", type=int, default=500000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--ckpt_every_step", type=int, default=10000)
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
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
    
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--perceptual_encoder", type=str, default="resnet", choices=['mlpmixer', 'biggan', 'resnet'])
    parser.add_argument(
        "--encoder_ckpt",
        default="",
        type=str,
        metavar="PATH",
        help="path to the encoder's checkpoint",
    )
    args = parser.parse_args()
    main(args)

    
    
    




    

