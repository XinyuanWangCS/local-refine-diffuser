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
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from einops.layers.torch import Rearrange
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from model_structures.biggan_classifier import BigGANClassifier

from model_structures.model_uncondition import DiT_Uncondition_models
from diffusion import create_diffusion
from model_structures.resnet import ResNet
from diffusers.models import AutoencoderKL

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


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        return x

class MLPMixerClassifier(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.mlp_mixer = MLPMixer(in_channels, dim, patch_size, image_size, depth, token_dim, channel_dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp_mixer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
    

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
        encoder.resnet.fc = nn.Linear(512, 2)
    else:
        raise ValueError(f'{args.encoder} is not supported.')
    del encoder_ckpt
    requires_grad(encoder, False)
    encoder.eval()
    
    model = DDP(model.to(device), device_ids=[rank]) # DataParrallel
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    g_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    d_opt = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay = 0)

    CELoss = torch.nn.CrossEntropyLoss()
    if args.resume:
         g_opt.load_state_dict(checkpoint["g_opt"])
    diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    

    if not args.resume:
        if rank == 0:
            dataset_name = args.data_path.split('/')[-1]
            experiment_index = len(glob(f"{args.results_dir}/{exp_name}-{dataset_name}*"))
            model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
            experiment_dir = f"{args.results_dir}/{exp_name}-{dataset_name}-{experiment_index:03d}--{model_string_name}"  # Create an experiment folder
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

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    #running_loss = 0 # the actual loss backward in generator
    #gen_running_loss = 0 # generator loss from DiT
    #g_adv_running_loss = 0 # adversarial loss
    dis_running_loss = 0 # sum loss
    start_time = time()

    tau = args.tau

    # Freeze all layers in DiT
    for param in model.parameters():
        param.requires_grad = False


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
            #dm_loss = loss_dict["loss"].mean()

            #print(f'pred_xt shape: {pred_xt.shape}')

            #preds = discriminator(pred_xt)
            #print(f'preds shape: {preds.shape}')
            #print(f'preds = {preds}')
            #print(f'gt shape: {gt.shape}')
            #print(f'gt = {gt}')
            #g_adv_loss = CELoss(preds, gt)
            

            #print(f'diffusion loss: {dm_loss}')
            #print(f'adversarial loss: {g_adv_loss}')

            # 0.8 tau (0.2 for g_adv_loss)
            #loss = tau * dm_loss + (1-tau) * g_adv_loss
            #g_opt.zero_grad()
            #loss.backward()
            #g_opt.step()
            #update_ema(ema, model.module)

            # Log loss values:
            #running_loss += loss.item()
            #gen_running_loss += dm_loss.item()
            #g_adv_running_loss += g_adv_loss.item()
            

            # -----------------
            #  train discriminator model 
            # -----------------
            
            #print(f'gt_xt = {gt_xt}, in shape of {gt_xt.shape}')
            #print(f'gt = {gt}, in shape of {gt.shape}')
            #print(f'discriminator processed gt_xt = {discriminator(pred_xt)}, in shape of {discriminator(pred_xt).shape}')
            real_loss = CELoss(discriminator(gt_xt), gt)
            print(f'real_loss = {real_loss}')
            fake_loss = CELoss(discriminator(pred_xt.detach()), fake)
            print(f'fake_loss = {fake_loss}')

            discriminator_loss = (real_loss + fake_loss) / 2
            #print(f'real loss: {real_loss}, fake loss: {fake_loss}, discriminator loss: {discriminator_loss}')
            

            d_opt.zero_grad()
            discriminator_loss.backward()
            d_opt.step()
            
            dis_running_loss += discriminator_loss.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                #avg_loss = torch.tensor(running_loss / log_steps, device=device)
                #dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                #avg_loss = avg_loss.item() / dist.get_world_size()

                #gen_avg_loss = torch.tensor(gen_running_loss / log_steps, device=device)
                #dist.all_reduce(gen_avg_loss, op=dist.ReduceOp.SUM)
                #gen_avg_loss = gen_avg_loss.item() / dist.get_world_size()

                #g_adv_avg_loss = torch.tensor(g_adv_running_loss / log_steps, device=device)
                #dist.all_reduce(g_adv_avg_loss, op=dist.ReduceOp.SUM)
                #g_adv_avg_loss = g_adv_avg_loss.item() / dist.get_world_size()

                dis_avg_loss = torch.tensor(dis_running_loss / log_steps, device=device)
                dist.all_reduce(dis_avg_loss, op=dist.ReduceOp.SUM)
                dis_avg_loss = dis_avg_loss.item() / dist.get_world_size()

                logger.info(f"(step={train_steps:07d}) Discriminator Loss: {dis_avg_loss: .4f} Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                #running_loss = 0
                #gen_running_loss = 0
                #g_adv_running_loss = 0
                dis_running_loss = 0

                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every_step == 0 or train_steps == args.total_steps -1 or train_steps==1:
                if rank == 0:
                    if args.use_ema:
                        #dit_checkpoint = {
                        #"model": model.module.state_dict(),
                        #"g_opt": g_opt.state_dict(),
                        #"epoch":epoch+1,
                        #"args": args,
                        #"experiment_dir":experiment_dir,
                        #"train_steps": train_steps,
                        #"ema": ema.state_dict(),
                    #}
                        encoder_checkpoint = {
                        "model": encoder.module.state_dict(),
                        "d_opt": d_opt.state_dict(),
                        "epoch":epoch+1,
                        "args": args,
                        "experiment_dir":experiment_dir,
                        "train_steps": train_steps
                    }
                    else:
                        #dit_checkpoint = {
                        #    "model": model.module.state_dict(),
                        #    "d_opt": d_opt.state_dict(),
                        #    "epoch":epoch+1,
                        #    "args": args,
                        #    "experiment_dir":experiment_dir,
                        #    "train_steps": train_steps,
                        #}
                        encoder_checkpoint = {
                        "model": encoder.module.state_dict(),
                        "d_opt": d_opt.state_dict(),
                        "epoch":epoch+1,
                        "args": args,
                        "experiment_dir":experiment_dir,
                        "train_steps": train_steps
                    }
                    #dit_checkpoint_dir = os.path.join(experiment_dir, #'dit_checkpoints')
                    #os.makedirs(dit_checkpoint_dir, exist_ok=True)
                    #dit_checkpoint_path_fin = os.path.join(dit_checkpoint_dir, #f'{train_steps:08d}.pt')
                    #torch.save(dit_checkpoint, dit_checkpoint_path_fin)
                    #logger.info(f"Saved checkpoint to {dit_checkpoint_path_fin}#")

                    encoder_checkpoint_dir = os.path.join(experiment_dir, 'encoder_checkpoints')
                    os.makedirs(encoder_checkpoint_dir, exist_ok=True)
                    encoder_checkpoint_path_fin = os.path.join(encoder_checkpoint_dir, f'{train_steps:08d}.pt')
                    torch.save(encoder_checkpoint, encoder_checkpoint_path_fin)
                    logger.info(f"Saved checkpoint to {encoder_checkpoint_path_fin}")

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
    parser.add_argument("--alpha", type=float, default=0.2)
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