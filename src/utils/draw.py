import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def annotate_points(x, y, ax, interval=1):
    min_val = min(y)
    for idx, (xi, yi) in enumerate(zip(x, y)):
        if yi == min_val:
            ax.scatter(xi, yi, color='red', zorder=3)
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', zorder=3)
        elif idx % interval == 0:
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center')
            ax.scatter(xi, yi, color='blue')


def main(args):
    file_dir = os.path.join(args.experiment_dir, args.xlsx_file)
    data = pd.read_excel(file_dir)

    x = data['ckpt_name']
    y1 = data['fid']
    y2 = data['kid']
    
    interval = max(1, len(x) // 10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y1, label='y1', marker='o')
    annotate_points(x, y1, ax, interval)
    ax.set_title('ckpt - fid data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    plt.savefig(os.path.join(args.experiment_dir, 'line_chart_fid.png'))
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y2, label='y2', marker='s')
    annotate_points(x, y2, ax, interval)
    ax.set_title('ckpt - kid data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    plt.savefig(os.path.join(args.experiment_dir, 'line_chart_kid.png'))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--xlsx_file", type=str, default='eval_scores.xlsx')
    args = parser.parse_args()
    main(args)