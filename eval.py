from cleanfid import fid
import os
from tqdm import tqdm
import argparse
from collections import defaultdict
import pandas as pd

def main(args):
    experiment_dir = args.experiment_dir
    #checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    sample_dir = f"{experiment_dir}/fid_samples"

    train_set_dir = args.train_set_dir
    results = []

    for ckpt_name in sorted(os.listdir(sample_dir), key=int): #digit names
        # Make a new directory inside the checkpoint's directory for the sampled images
        sample_dir_path = os.path.join(sample_dir, ckpt_name) #e.g. "./results/001-DiT-S-4/checkpoints/0000001/sampled_images"
        print('------------------------------------------------------------')
        print(f'Evaluating: {sample_dir_path}')
        fid_score = fid.compute_fid(train_set_dir, sample_dir_path, mode = 'clean')
        kid_score = fid.compute_kid(train_set_dir, sample_dir_path, mode = 'clean')

        results.append({'ckpt_name':ckpt_name, 'fid':fid_score, 'kid':kid_score})
        print(f'fid: {fid_score} kid: {kid_score}')
    results = pd.DataFrame(results)
    results.to_excel(os.path.join(experiment_dir, 'eval_scores.xlsx'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--train_set_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
