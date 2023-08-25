import torch
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
from PIL import Image
from glob import glob
import argparse
import logging
import os
import time
from model_structures.model_uncondition import DiT_Uncondition_models
from diffusion import create_diffusion
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
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_Uncondition_models[args.model]( 
        input_size=latent_size
    ).to(device)
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    train_steps = 0
    # Resume training: continue ckpt args.resume
    if args.resume:
        if not os.path.isfile(args.resume):
            raise ValueError(f'checkpoint dir not exist: {args.resume}')
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{device}'))
        
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        
        print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]))
        experiment_dir = checkpoint["experiment_dir"]
        train_steps = checkpoint["train_steps"]
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
    else:
        print(f'Build model: {args.model}')

    # DataParrallel
    model = DDP(model, device_ids=[rank]) 
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    d_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if args.resume:
        d_opt.load_state_dict(checkpoint["d_opt"])
        checkpoint = None 
        time.sleep(3)
    diffusion = create_diffusion(str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule
    
    # Setup an experiment folder:
    exp_name = args.experiment_name
    if not args.resume:
        if rank == 0:
            dataset_name = args.data_path.split('/')[-1]
            experiment_index = len(glob(f"{args.results_dir}/{exp_name}-{dataset_name}*"))
            model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
            experiment_dir = f"{args.results_dir}/{exp_name}-{dataset_name}-{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
            os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            os.makedirs(experiment_dir, exist_ok=True)

            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")
            logger.info("Arguments:")
            for k, v in vars(args).items():
                logger.info(f'{k}: {v}')
        else:
            logger = create_logger(None)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
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

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    log_steps = 0
    d_running_loss = 0
    start_time = time.time()

    if rank != 0 and not args.resume:
        experiment_index = len(glob(f"{args.results_dir}/{exp_name}*"))
        dataset_name = args.data_path.split('/')[-1]
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{exp_name}-{experiment_index:03d}-{dataset_name}--{model_string_name}"  # Create an experiment folder

    logger.info(f"Total epoch number: {args.epochs}.")
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        logger.info(f"Begin epoch: {epoch}")
        for x, _ in loader:
            x = x.to(device)

            # train diffusion model 
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict() #y=y
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            dm_loss = loss_dict["loss"].mean()
            
            d_loss = dm_loss
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

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
        if epoch % args.ckpt_every == 0 or epoch == args.epochs -1:
            if rank == 0:
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
                checkpoint_path_fin = os.path.join(checkpoint_dir, f'{epoch:07d}.pt')
                torch.save(checkpoint, checkpoint_path_fin)
                logger.info(f"Saved checkpoint to {checkpoint_path_fin}")
            dist.barrier()

    logger.info("Done!")
    cleanup()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="baseline")
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
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--continue_training", type=bool, default=False)
    parser.add_argument("--continue_ckpt_dir", type=str, default='')
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    args = parser.parse_args()
    main(args)
