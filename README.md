# Local Refine Diffuser

## This project aims to improve the details of diffution models.

### Install the environment:
```bash
conda env create -f environment.yml
conda activate DiT
```

### Train baseline DiT:
GPU: Nvidia A4500 10G * 4
example: use DiT_Uncondition-B/4 backbone and celebhq256 dataset
```bash
torchrun --nnodes=1 --nproc_per_node=4 src/train_baseline_dit.py --model DiT_Uncondition-B/4 --data_path datasets/celebahq256 --image-size 256 --total_steps 400000 --ckpt_every_step 10000  --global-batch-size 128 --use_ema True 
```
Resume checkpoint:
```bash
torchrun --nnodes=1 --nproc_per_node=4 src/train_baseline_dit.py --model DiT_Uncondition-B/4 --data_path datasets/celebahq256 --image-size 256 --total_steps 400000 --ckpt_every_step 10000  --global-batch-size 128 --use_ema True --resume results/baseline-celebahq256-000-DiT_Uncondition-B-4/checkpoints/00200000.pt
```

### Sample images for fid evaluation:
```bash
torchrun --nnodes=1 --nproc_per_node=4 src/fid_sample.py --experiment_dir results/perceptual-celebahq256old-001-DiT_Uncondition-B-4 --model DiT_Uncondition-B/4 --fid_samples 5000 --image-size 256 --global-batch-size 128 --num_sampling_steps 1000 --use_ema True
```

### Evaluate trained checkpoins:
example: the trained example above and lfw dataset
```bash
python src/eval.py --experiment_dir results/perceptual_end_to_end-celebahq256-004-DiT_Uncondition-B-4/ --train_set_dir datasets/celebahq256/ --sample_folder_name fid_samples --output_file_name 'eval_scores.xlsx' --cal_kid False
```

### Sample images from a DiT model.   
from num_sampling_steps to end_step
```bash
torchrun --nnodes=1 --nproc_per_node=8 src/sample_one_model.py --checkpoint_dir results/baseline-celebahq256-000-DiT_Uncondition-B-4/checkpoints/00200000.pt --save_dir celebahq256_50step --model DiT_Uncondition-B/4 --fid_samples 30000 --image-size 256 --global-batch-size 128 --num_sampling_steps 1000 --use_ema True --end_step 50
```


# GAN

### train_discriminator.py
```bash
torchrun --nnodes=1 --nproc_per_node=8 src/train_discriminator.py --model resnet --data_path datasets/gan_data/ --image_size 256 --epochs 1000 --global_batch_size 256 --log_every_step 100 --ckpt_every_epoch 10
```

### train_gan_end_to_end.py
use pretrained discriminator to provide gan_loss. The pretrained discriminator is frozen. Random sample t from 0 ~ 50
```bash
torchrun --nnodes=1 --nproc_per_node=8 src/train_gan_end_to_end.py --data_path datasets/celebahq256/ --image_size 256 --total_steps 100000 --global_batch_size 128 --log_every_step 50 --ckpt_every_step 10000 --num_sampling_steps 1000 --start_step 50 --load_ema False --resume results/baseline-celebahq256-000-DiT_Uncondition-B-4/checkpoints/00200000.pt --model DiT_Uncondition-B/4 --alpha 0.2 --discriminator resnet --discriminator_ckpt results/pretrain_discriminator-resnet-gan_data-000/checkpoints/00000049.pt
```

### Draw examples
```bash
python src/utils/draw_examples.py --experiment_dir results/baseline-ffhq5k-000--DiT_Uncondition-S-4/ --image_num 12 --target_dir results/baseline-ffhq5k-000--DiT_Uncondition-S-4/plot_examples
```
### Train ImageNet Classifier for perceptual loss
```bash
torchrun --nnodes=1 --nproc_per_node=8  src/train_imagenet_classifier.py --model biggan --data_path dataset/imagenet1k --image_size 256 --epochs 200 --global-batch-size 256 --log-every 50 --ckpt-every 1 --test-every-epoch 1 --use_ema True
```

### Perceptual end to end noise
```bash
torchrun --nnodes=1 --nproc_per_node=8 src/train_dit_end_to_end_perceptual_use_noise.py --model DiT_Uncondition-B/4 --data_path datasets/celebahq256/  --image_size 256 --total_steps 10000 --ckpt_every_step 500 --global_batch_size 128 --load_ema False --start_step 1000 --perceptual_encoder resnet --encoder_ckpt encoder_ckpts/resnet00000070.pt --resume results/baseline-celebahq256-000-DiT_Uncondition-B-4/checkpoints/00200000.pt --alpha 1.5
```

```bash
torchrun --nnodes=1 --nproc_per_node=8 src/train_dit_end_to_end_perceptual_use_noise_new_p_loss.py --model DiT_Uncondition-B/4 --data_path datasets/celebahq256/  --image_size 256 --total_steps 10000 --ckpt_every_step 500 --global_batch_size 128 --load_ema False --start_step 1000 --perceptual_encoder resnet --encoder_ckpt encoder_ckpts/resnet00000070.pt --resume results/baseline-celebahq256-000-DiT_Uncondition-B-4/checkpoints/00200000.pt --alpha 0.1
```
CUDA_VISIBLE_DEVICES=


### test
```bash
torchrun --nnodes=1 --nproc_per_node=8 src/sample_t_sequence.py --checkpoint_dir results/baseline-celebahq256-000-DiT_Uncondition-B-4/checkpoints/00180000.pt --save_dir results/test_tiff --fid_samples 100 --end_step 0 
```

### Sample t sequence
```bash
torchrun --nnodes=1 --nproc_per_node=4 src/sample_t_sequence.py --checkpoint_dir results/baseline-celebahq256-000-DiT_Uncondition-B-4/checkpoints/00200000.pt --num_samples 128 --start_t 0 --end_t 1000 --interval 100 --load_ema False --use_seed True
```

# gan_pipeline_end_to_end.py
```bash
torchrun --nnodes=1 --nproc_per_node=4 src/gan_pipeline_end_to_end.py --model DiT_Uncondition-B/4 --data_dir datasets/celebahq256/  --image_size 256 --total_steps 10000 --dis_total_steps 1500 --global_batch_size 128 --ckpt_every_step 5000 --iteration_num 10 --num_samples 1280 --start_t 0 --end_t 50 --interval 1 --discriminator condition_resnet --resume results/baseline-celebahq256-000-DiT_Uncondition-B-4/checkpoints/00200000.pt --alpha 0.025
```

torchrun --nnodes=1 --nproc_per_node=4 src/gd_sample_one_model_one_t.py --checkpoint_dir pretrained_models/256x256_diffusion_uncond.pt

### Memory requirement:
DiT_Uncondition-S-4:6242
DiT_Uncondition-B-4: 12178

With CLIP encoder:
batch_size 1 per GPU: 5516
batch_size 2 per GPU: 8244
batch_size 6 per GPU: 19192

DiT baseline: 4 GPU 384 resume 320


```bash
CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --nproc_per_node=1 --master_port 29502 src/fid_sample.py --experiment_dir results/perceptual_end_to_end-celebahq256-005-DiT_Uncondition-B-4/ --ckpt_folder epoch_checkpoints --save_dir epoch_fid_samples --model DiT_Uncondition-B/4 --fid_samples 100 --image-size 256 --global-batch-size 128 --num_sampling_steps 1000 --use_ema False
```