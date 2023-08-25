import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_images_from_dir(directory, image_num, save_path):
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]
    image_files = sorted(image_files)[:image_num]
    
    rows = (image_num + 3) // 4
    fig, axs = plt.subplots(rows, 4, figsize=(5*4, 5*rows))
    
    for idx, ax in enumerate(axs.ravel()):
        if idx < len(image_files):
            img = mpimg.imread(image_files[idx])
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
    
    # Ensure no spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main(experiment_dir, image_num, target_dir=None):
    if target_dir is None:
        target_dir = experiment_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for subdir in sorted(os.listdir(experiment_dir)):
        full_subdir_path = os.path.join(experiment_dir, subdir)
        if os.path.isdir(full_subdir_path):
            save_path = os.path.join(target_dir, f"{subdir}.png")
            plot_images_from_dir(full_subdir_path, image_num, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--image_num", type=int, required=True)
    parser.add_argument("--target_dir", type=str, default=None)
    args = parser.parse_args()
    main(args.experiment_dir, args.image_num, args.target_dir)