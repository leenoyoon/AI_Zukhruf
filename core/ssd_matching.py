import numpy as np
from numba import njit

@njit
def calculate_ssd_manual(overlap_existing, overlap_candidate):
    rows, cols = overlap_existing.shape
    error = 0.0
    for i in range(rows):
        for j in range(cols):   
            diff = float(overlap_existing[i, j]) - float(overlap_candidate[i, j])
            error += diff * diff
    return error

@njit
def find_best_matching_patch(patch_bank, overlap_top, overlap_left, overlap_width, mode):
    num_patches = patch_bank.shape[0]
    patch_size = patch_bank.shape[1]
    errors = np.zeros(num_patches, dtype=np.float64)
    for i in range(num_patches):
        patch = patch_bank[i]
        err = 0.0
        if mode == 1 or mode == 3:
            candidate_left = patch[:, :overlap_width]
            err += calculate_ssd_manual(overlap_left, candidate_left)  
        
        if mode == 2 or mode == 3:
            candidate_top = patch[:overlap_width, :]
            err += calculate_ssd_manual(overlap_top, candidate_top)
            
        errors[i] = err
        
    return errors

def generate_texture_canvas(patch_bank, seed_patch, canvas_rows, canvas_cols, patch_size, overlap_width):
    h_pixels = canvas_rows * patch_size - (canvas_rows - 1) * overlap_width
    w_pixels = canvas_cols * patch_size - (canvas_cols - 1) * overlap_width
    canvas = np.zeros((h_pixels, w_pixels), dtype=np.uint8)
    canvas[:patch_size, :patch_size] = seed_patch
    dummy_array = np.zeros((1, 1), dtype=np.uint8)
    for r in range(canvas_rows):
        for c in range(canvas_cols):
            if r == 0 and c == 0: 
                continue 
                
            y = r * (patch_size - overlap_width)
            x = c * (patch_size - overlap_width)
    
            overlap_top = dummy_array
            overlap_left = dummy_array
            mode = 0
            
            if r > 0 and c > 0: 
                overlap_top = canvas[y:y+overlap_width, x:x+patch_size]
                overlap_left = canvas[y:y+patch_size, x:x+overlap_width]
                mode = 3
            elif r > 0: 
                overlap_top = canvas[y:y+overlap_width, x:x+patch_size]
                mode = 2
            else: 
                overlap_left = canvas[y:y+patch_size, x:x+overlap_width]
                mode = 1
            
            errors = find_best_matching_patch(patch_bank, overlap_top, overlap_left, overlap_width, mode)
            
            min_err = np.min(errors)
            tolerance = 0.1 
            threshold = min_err * (1.0 + tolerance)
            
            best_indices = np.where(errors <= threshold)[0] 
            chosen_idx = np.random.choice(best_indices)
            
            canvas[y:y+patch_size, x:x+patch_size] = patch_bank[chosen_idx]
        
    return canvas