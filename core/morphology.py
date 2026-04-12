import numpy as np
from numba import njit

@njit
def dilation(binary_image, k=2):
    rows, cols = binary_image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            max_val = 0
            for ki in range(-k, k + 1):
                for kj in range(-k, k + 1):
                    ni, nj = i + ki, j + kj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if binary_image[ni, nj] == 255:
                            max_val = 255
            result[i, j] = max_val
    return result

@njit
def erosion(binary_image, k=2):
    rows, cols = binary_image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            min_val = 255
            for ki in range(-k, k + 1):
                for kj in range(-k, k + 1):
                    ni, nj = i + ki, j + kj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if binary_image[ni, nj] == 0:
                            min_val = 0
            result[i, j] = min_val
    return result

@njit
def morphological_close(binary_image):
    dilated = dilation(binary_image, k=2)
    closed = erosion(dilated, k=2)
    return closed