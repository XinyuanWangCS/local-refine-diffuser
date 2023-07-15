from cleanfid import fid
import os
import logging
import shutil
import random

from model_uncondition import DiT_Uncondition

checkpoint_dir = "./results/000-DiT-S-4/checkpoints/"
sample_dir = './sample_img/'
train_set = './dataset_img'

#model = DiT_Uncondition(depth=12, hidden_size=384, patch_size=4, num_heads=6,)

for filename in sorted(os.listdir(checkpoint_dir), key=int): #digit names
    new_file_path = f"{checkpoint_dir}/{filename}" #e.g. "./results/001-DiT-S-4/checkpoints/0000001"

    # Make a new directory inside the checkpoint's directory for the sampled images
    image_dir_path = os.path.join(new_file_path, 'sampled_images') #e.g. "./results/001-DiT-S-4/checkpoints/0000001/sampled_images"
    os.makedirs(image_dir_path, exist_ok=True)

    sample_dir_path = os.path.join(sample_dir, filename) #e.g. "./sample_img/0000001"
    
    if os.path.isdir(sample_dir_path):
        all_files = [f for f in os.listdir(sample_dir_path) if f.endswith('.png')]
        selected_files = random.sample(all_files, 50)
        for file_name in selected_files: #.png names
            src_file_path = os.path.join(sample_dir_path, file_name) #e.g. ""./sample_img/0000001/0000001.png"
            dest_file_path = os.path.join(image_dir_path, file_name) #e.g. "./results/001-DiT-S-4/checkpoints/0000001/sampled_images/0000001.png"
            shutil.copy(src_file_path, dest_file_path)
    
    print(f'done for {filename}')


    fid_score = fid.compute_fid(train_set, f'{sample_dir}/{filename}', mode = 'clean')
    kid_score = fid.compute_kid(train_set, f'{sample_dir}/{filename}', mode = 'clean')

    # Set up logger
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(new_file_path, 'scores.log'))
    logger.addHandler(handler)

    # Log the fid_score and kid_score
    logger.info(f'FID score: {fid_score}')
    logger.info(f'KID score: {kid_score}')

    # Remove the handler at the end
    logger.removeHandler(handler)
