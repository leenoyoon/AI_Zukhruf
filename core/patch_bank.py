import numpy as np
import math 
from numba import njit


def build_patch_bank(image, patch_size=32, overlap_ratio=0.25):
    step = int(patch_size * (1 - overlap_ratio))
    h, w = image.shape
    shape = ((h - patch_size) // step + 1, (w - patch_size) // step + 1, patch_size, patch_size)
    strides = (step * image.strides[0], step * image.strides[1], image.strides[0], image.strides[1]) 
    patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    return patches.reshape(-1, patch_size, patch_size)


@njit
def get_best_initial_patch(patch_bank):
    num_patches, rows, cols = patch_bank.shape
    N = rows * cols
    max_std = -1.0
    best_index = 0
    for p in range(num_patches):
        patch = patch_bank[p]
        sum_val = 0.0
        for i in range(rows):
            for j in range(cols):
                sum_val += patch[i, j]
        mean = sum_val / N
        
        variance_sum = 0.0
        for i in range(rows):
            for j in range(cols):
                variance_sum += (patch[i, j] - mean) ** 2
        variance = variance_sum / N
        
        std_dev = math.sqrt(variance)

        if std_dev > max_std:
            max_std = std_dev
            best_index = p
       
    return best_index, patch_bank[best_index], max_std