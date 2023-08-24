# Local Refine Diffuser

This project aims to improve the details of diffution models.

Install the environment:
```bash
conda env create -f environment.yml
conda activate DiT
```

Train baseline and sample images for fid, kid calculation:
GPU: 4 20G GPUs
example: use DiT_Uncondition-B/4 backbone and ffhq1k dataset
```bash
torchrun --nnodes=1 --nproc_per_node=4 src/train_baseline_dit.py --model DiT_Uncondition-B/4 --data_path dataset/images/ffhq1k --epochs 15000 --ckpt_every 500 --image-size 256 --global-batch-size 276
```

```bash
torchrun --nnodes=1 --nproc_per_node=4 src/train_baseline_dit.py --model DiT_Uncondition-S/4 --data_path dataset/images/ffhq3k --epochs 15000 --ckpt_every 500 --image-size 256 --global-batch-size 384
```

Train DiT with CLIP encoder perceptual loss:
```bash
torchrun --nnodes=1 --nproc_per_node=2 src/train_with_encoder.py --model DiT_Uncondition-S/4 --data_path dataset/images/lfw_funneled --epochs 100 --ckpt_every 10 --image-size 256 --global-batch-size 12
```

Train DiT with ResNet50 encoder perceptual loss:
```bash
!torchrun --nnodes=1 --nproc_per_node=4 src/train_with_resnet.py --model DiT_Uncondition-S/4 --data_path dataset/images/lfw_funneled --epochs 200 --ckpt_every 10 --image-size 256 --global-batch-size 24
```


Sample images for fid evaluation:
```bash
torchrun --nnodes=1 --nproc_per_node=4 src/fid_sample.py --experiment_dir results/baseline-001-ffhq1k--DiT_Uncondition-S-4 --model DiT_Uncondition-S/4 --fid_samples 3000 --image-size 256 --global-batch-size 128 --num_sampling_steps 1000
```

Evaluate trained checkpoins:
example: the trained example above and lfw dataset
```bash
python src/eval.py --experiment_dir results/test --train_set_dir dataset/images/ffhq1k
```

Memory requirement:
DiT_Uncondition-S-4:6242
DiT_Uncondition-B-4: 12178

With CLIP encoder:
batch_size 1 per GPU: 5516
batch_size 2 per GPU: 8244
batch_size 6 per GPU: 19192

DiT baseline: 4 GPU 384