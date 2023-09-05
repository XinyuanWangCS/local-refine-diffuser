# Local Refine Diffuser

## This project aims to improve the details of diffution models.

### Install the environment:
```bash
conda env create -f environment.yml
conda activate DiT
```

### Train baseline DiT:
GPU: Nvidia A4500 10G * 4
example: use DiT_Uncondition-B/4 backbone and ffhq1k dataset
```bash
torchrun --nnodes=1 --nproc_per_node=4 src/train_baseline_dit.py --model DiT_Uncondition-B/4 --data_path dataset/ffhq256 --image-size 256 --total_steps 500000 --ckpt_every_step 10000  --global-batch-size 256 --use_ema True
```

### Train DiT with ResNet perceptual loss:
```bash
torchrun --nnodes=1 --nproc_per_node=8 src/train_dit_perceptual_loss.py --model DiT_Uncondition-B/4 --data_path datasets/celebahq256old/  --image-size 256 --total_steps 233770 --ckpt_every_step 11500 --global-batch-size 128 --use_ema True --perceptual_encoder resnet --encoder_ckpt encoder_ckpts/resnet00000070.pt --resume pretrained_models/DiT-B-4-celebahqold256/0001500.pt --alpha 2
```

### Sample images for fid evaluation:
```bash
torchrun --nnodes=1 --nproc_per_node=4 src/fid_sample.py --experiment_dir results/baseline-001-ffhq1k--DiT_Uncondition-S-4 --model DiT_Uncondition-S/4 --fid_samples 3000 --image-size 256 --global-batch-size 128 --num_sampling_steps 1000 --use_ema True
```

### Evaluate trained checkpoins:
example: the trained example above and lfw dataset
```bash
python src/eval.py --experiment_dir results/test --train_set_dir dataset/images/ffhq1k --folder_name fid_samples --file_name 'eval_scores.xlsx' --cal_kid False
```

### Draw examples
```bash
python src/utils/draw_examples.py --experiment_dir results/baseline-ffhq5k-000--DiT_Uncondition-S-4/examples/ --image_num 8 --target_dir results/baseline-ffhq5k-000--DiT_Uncondition-S-4/plot_examples
```
### Train ImageNet Classifier for perceptual loss
```bash
torchrun --nnodes=1 --nproc_per_node=8  src/train_imagenet_classifier.py --model biggan --data_path dataset/imagenet1k --image_size 256 --epochs 200 --global-batch-size 256 --log-every 50 --ckpt-every 1 --test-every-epoch 1 --use_ema True
```

### Memory requirement:
DiT_Uncondition-S-4:6242
DiT_Uncondition-B-4: 12178

With CLIP encoder:
batch_size 1 per GPU: 5516
batch_size 2 per GPU: 8244
batch_size 6 per GPU: 19192

DiT baseline: 4 GPU 384 resume 320