import torch
import torch.nn.functional as F
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from PIL import Image
from glob import glob
from collections import OrderedDict
import argparse
import logging
import os
import time
from model_structures.model_uncondition import DiT_Uncondition_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from copy import deepcopy
from utils.contrastive_learning_dataset import ContrastiveLearningDataset
from utils.utils import str2bool, requires_grad, build_logger

    
#################################################################################
#                             Training Helper Functions                         #
#################################################################################
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

#################################################################################
#                                  Training Loop                                #
#################################################################################

def info_nce_loss(features, batch_size, device, temperature=0.07):
    #print(features)
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    #print(similarity_matrix)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels
    
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
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_Uncondition_models[args.model]( 
        input_size=latent_size
    )

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
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cpu'))
        
        args.start_epoch = checkpoint["epoch"]
        if args.use_ema:
            model.load_state_dict(checkpoint["ema"])
            print("=> loaded ema checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]))
        else:
            model.load_state_dict(checkpoint["model"])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]))
        ema.load_state_dict(checkpoint["ema"])
        train_steps = checkpoint["train_steps"]
        del checkpoint
    else:
        print(f'Build model: {args.model}')

    # DataParrallel
    model = DDP(model.to(device), device_ids=[rank]) 
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    d_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    
    # Setup an experiment folder:
    exp_name = args.experiment_name
    
    experiment_dir, logger = build_logger(args, rank)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Setup data
    batch_size = int(args.global_batch_size // dist.get_world_size())
    dataset = ContrastiveLearningDataset(args.data_dir, image_size=args.image_size).get_dataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed 
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_dir})")

    # Variables for monitoring/logging purposes:
    log_steps = 0
    d_running_loss = 0
    start_time = time.time()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    '''for param in model.parameters():
        print(param)'''
    logger.info(f"Total step number: {args.total_steps}.")
    for epoch in range(args.start_epoch, 100000):
        model.train()
        sampler.set_epoch(epoch)
        logger.info(f"Begin epoch: {epoch}")
        for x, _ in loader:
            t = torch.randint(0, diffusion.num_timesteps, (4,), device=device)# TODO: 
            t = t.repeat(batch_size // 4)
            t = t.repeat(2)
            x = torch.cat(x, dim=0)
            x = x.to(device)
            #print(x.shape)
            # train diffusion model 
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            loss_dict = diffusion.training_losses_noise_output(model, x, t)
            features = loss_dict["pred"]
            features = features.view(features.shape[0], -1)
            logits, labels = info_nce_loss(features, batch_size=batch_size, device=device)
            
            d_loss = criterion(logits, labels)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()
            if args.use_ema:
                update_ema(ema, model.module)

            # Log loss values:
            d_running_loss += d_loss.item()

            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                d_avg_loss = torch.tensor(d_running_loss / log_steps, device=device)
                dist.all_reduce(d_avg_loss, op=dist.ReduceOp.SUM)
                d_avg_loss = d_avg_loss.item() / dist.get_world_size()

                logger.info(f"(step={train_steps:07d}) D_Train Loss: {d_avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                d_running_loss = 0
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
                break
        if train_steps >= args.total_steps:
            return
        dist.barrier()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="baseline")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_Uncondition_models.keys()), default="DiT_Uncondition-B/4")
    parser.add_argument("--image_size", type=int, choices=[32, 64, 128, 224, 256, 512], default=256)
    parser.add_argument("--total_steps", type=int, default=500000)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num_workers", type=int, default=4)
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
    args = parser.parse_args()
    main(args)
