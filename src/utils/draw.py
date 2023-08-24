import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os



def main(args):
    file_dir = os.path.join(args.experiment_dir, args.xlsx_file)
    data = pd.read_excel(file_dir)

    x = data['ckpt_name']
    y1 = data['fid']
    y2 = data['kid']

    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, label='y1', marker='o')
    plt.title('ckpt - fid data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.experiment_dir, 'line_chart_fid.png'))  # 保存图表为图片文件
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(x, y2, label='y2', marker='s')
    plt.title('ckpt - kid data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.experiment_dir, 'line_chart_kid.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--xlsx_file", type=str, default='eval_scores.xlsx')
    args = parser.parse_args()
    main(args)