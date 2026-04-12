import numpy as np
from numba import njit

@njit
def rgb_to_grayscale(image_array):
    rows = image_array.shape[0]  
    cols = image_array.shape[1]
    gray_image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            b, g, r = image_array[i, j, 0], image_array[i, j, 1], image_array[i, j, 2]
            gray_val = (0.2989 * r) + (0.5870 * g) + (0.1140 * b)
            gray_image[i, j] = int(gray_val)
    return gray_image

@njit
def histogram_equalization(gray_image):
    rows, cols = gray_image.shape
    total_pixels = rows * cols
    
    hist = np.zeros(256, dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            pixel_val = gray_image[i, j]
            hist[pixel_val] += 1
            
    cdf = np.zeros(256, dtype=np.int32)
    cdf[0] = hist[0]  
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
        
    cdf_min = 0
    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break
            
    new_values = np.zeros(256, dtype=np.uint8)
    for v in range(256):
        numerator = cdf[v] - cdf_min
        denominator = total_pixels - cdf_min
        if denominator == 0:
            new_values[v] = v
        else:
            val = (numerator / denominator) * 255
            new_values[v] = np.uint8(round(val))
            
    equalized_image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            equalized_image[i, j] = new_values[gray_image[i, j]]
            
    return equalized_image


@njit
def otsu_threshold(gray_image):
    rows, cols = gray_image.shape
    total_pixels = rows * cols
    
    hist = np.zeros(256, dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            hist[gray_image[i, j]] += 1
            
    best_threshold = 0
    max_variance = 0.0
    
    for t in range(256):
        bg_pixels = 0.0
        bg_sum = 0.0
        for i in range(0, t):
            bg_pixels += hist[i]
            bg_sum += i * hist[i]
            
        fg_pixels = total_pixels - bg_pixels
        
        if bg_pixels == 0 or fg_pixels == 0:
            continue
            
        w_bg = bg_pixels / total_pixels
        w_fg = fg_pixels / total_pixels
        mu_bg = bg_sum / bg_pixels
        
        fg_sum = 0.0
        for i in range(t, 256):
            fg_sum += i * hist[i]
        mu_fg = fg_sum / fg_pixels
        
        variance = w_bg * w_fg * (mu_bg - mu_fg) ** 2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
            
    binary_mask = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if gray_image[i, j] > best_threshold:
                binary_mask[i, j] = 255
            else:
                binary_mask[i, j] = 0
                
    return binary_mask, best_threshold

@njit
def apply_mask(equalized_image, binary_mask):
    rows, cols = equalized_image.shape
    final_map = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if binary_mask[i, j] == 255:
                final_map[i, j] = equalized_image[i, j]
            else:
                final_map[i, j] = 0
    return final_map