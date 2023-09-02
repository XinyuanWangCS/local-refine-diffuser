from cleanfid import fid
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def annotate_points(x, y, ax, interval=1):
    min_val = min(y)
    for idx, (xi, yi) in enumerate(zip(x, y)):
        if yi == min_val:
            ax.scatter(xi, yi, color='red', zorder=3)
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', zorder=3)
        elif idx % interval == 0:
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center')
            ax.scatter(xi, yi, color='blue')
            
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot(args, experiment_dir, file_name):
    file_dir = os.path.join(experiment_dir, file_name)
    data = pd.read_excel(file_dir)

    x = data['ckpt_name']
    y1 = data['fid']
    
    
    interval = max(1, len(x) // 10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y1, label='y1', marker='o')
    annotate_points(x, y1, ax, interval)
    ax.set_title('ckpt - fid data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    plt.savefig(os.path.join(experiment_dir, 'line_chart_fid.png'))
    plt.show()

    if args.cal_kid:
        y2 = data['kid']
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y2, label='y2', marker='s')
        annotate_points(x, y2, ax, interval)
        ax.set_title('ckpt - kid data')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()
        plt.savefig(os.path.join(experiment_dir, 'line_chart_kid.png'))
        plt.show()

def main(args):
    experiment_dir = args.experiment_dir
    #checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    sample_dir = f"{experiment_dir}"
    sample_dir = os.path.join(sample_dir, args.folder_name)

    train_set_dir = args.train_set_dir
    results = []

    for ckpt_name in sorted(os.listdir(sample_dir), key=int): #digit names
        # Make a new directory inside the checkpoint's directory for the sampled images
        sample_dir_path = os.path.join(sample_dir, ckpt_name) #e.g. "./results/001-DiT-S-4/checkpoints/0000001/sampled_images"
        print('------------------------------------------------------------')
        print(f'Evaluating: {sample_dir_path}')
        fid_score = fid.compute_fid(train_set_dir, sample_dir_path, mode = 'clean')
        if args.cal_kid:
            kid_score = fid.compute_kid(train_set_dir, sample_dir_path, mode = 'clean')
            results.append({'ckpt_name':ckpt_name, 'fid':fid_score, 'kid':kid_score})
            print(f'fid: {fid_score} kid: {kid_score}')
        else:
            results.append({'ckpt_name':ckpt_name, 'fid':fid_score})
            print(f'fid: {fid_score}')
    results = pd.DataFrame(results)
    results.to_excel(os.path.join(experiment_dir, args.file_name))
    plot(args=args, experiment_dir=experiment_dir, file_name=args.file_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, default='fid_samples')
    parser.add_argument("--file_name", type=str, default='eval_scores.xlsx')
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--train_set_dir", type=str, required=True)
    parser.add_argument("--cal_kid", type=str2bool, default=False)
    args = parser.parse_args()
    main(args)
