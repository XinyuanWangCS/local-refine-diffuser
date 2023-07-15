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
torchrun --nnodes=1 --nproc_per_node=4 train_baseline_with_eval.py --model DiT_Uncondition-B/4 --data_path dataset/images/lfw --epochs 2000 --ckpt_every 50 --fid_samples 5000
```

Evaluate trained checkpoins:
example: the trained example above and lfw dataset
```bash
python eval.py --experiment_dir results/001-DiT_Uncondition-B-4 --train_set_dir dataset/images/lfw
```

Memory requirement:
DiT_Uncondition-B-4: 12178