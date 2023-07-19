# Local Refine Diffuser

This project aims to improve the details of diffution models.

Install the environment:
```bash
conda env create -f environment.yml
conda activate DiT
```

Train baseline and sample images for fid, kid calculation:
example: use DiT_Uncondition-B/4 backbone and lfw dataset
```bash
torchrun --nnodes=1 --nproc_per_node=4 train_baseline_with_eval.py --model DiT_Uncondition-S/4 --data_path dataset/images/lfw --epochs 1500 --ckpt_every 100 --fid_samples 5000 --image-size 224 --global-batch-size 256
```
```bash
torchrun --nnodes=1 --nproc_per_node=3 train_baseline_with_eval.py --model DiT_Uncondition-S/4 --data_path dataset/images/wiki --epochs 1000 --ckpt_every 100 --fid_samples 10000 --image-size 224 --global-batch-size 384
```
Evaluate trained checkpoins:
example: the trained example above and lfw dataset
```bash
python eval.py --experiment_dir results/lfw-000-DiT_Uncondition-S-4 --train_set_dir dataset/images/lfw
```

Sample images for fid evaluation:
```
torchrun --nnodes=1 --nproc_per_node=3 fid_sample.py --experiment_dir results/lfw_funneled-004-DiT_Uncondition-S-4-with_encoder --model DiT_Uncondition-S/4 --fid_samples 6000 --image-size 224 --global-batch-size 192 --num_sampling_steps 250
```

Memory requirement:
DiT_Uncondition-S-4:6242
DiT_Uncondition-B-4: 12178

With encoder:
batch_size 1: 5516
batch_size 2: 8244
batch_size 6: 19192