from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

def batch_update_statistics(old_mean, old_variance, old_count, new_data):
    n = old_count
    m = len(new_data)
    y_mean = np.mean(new_data)
    y_variance = np.var(new_data, ddof=1)  # ddof=1 for unbiased variance

    new_mean = (n * old_mean + m * y_mean) / (n + m)
    new_variance = ((n - 1) * old_variance + (m - 1) * y_variance + n * (old_mean - new_mean)**2 + m * (y_mean - new_mean)**2) / (n + m - 1)

    return new_mean, new_variance, n + m

img_list = []
def calculate_mean_std(img_dir):
    img_paths = glob(img_dir+'/*/*/*.jpg')
    count = 0
    std = 0
    mean = 0
    for p in tqdm(img_paths):
        img = Image.open(p).convert('RGB')
        img_arr = np.array(img).astype(np.float32) / 255 #[h, w, 3]
        img_list.append(img_arr.reshape(-1,3))
        count += 1
        # if count >= 2:
        #     break

    imgs = np.concatenate(img_list)
    mean = imgs.mean(axis=0)
    print(mean)
    std = imgs.std(axis=0)
    print(std)
    # print(f'mean={mean:.3f}, std={std:.3f}')

calculate_mean_std('/home/zhougaowei/datasets/xray/mvtec/cans')

"""
mean = [0.03016077, 0.03016077, 0.03016077]
std = [0.17366856, 0.17366856, 0.17366856]
"""