import cv2

from core.image_processing import (
    rgb_to_grayscale,
    histogram_equalization,
    otsu_threshold,
    apply_mask
)
from core.morphology import morphological_close
from core.patch_bank import (
    build_patch_bank,
    get_best_initial_patch
)

if __name__ == "__main__":
    image_path = r"C:\Users\ASUS\Desktop\pattern1.jpg" 
    original_img = cv2.imread(image_path)
    
    if original_img is not None:
        original_img = cv2.resize(original_img, (512, 512))
        
        gray_img = rgb_to_grayscale(original_img)
        equalized_img = histogram_equalization(gray_img)
        mask_img, optimal_thresh = otsu_threshold(gray_img)
        smooth_mask = morphological_close(mask_img)
        final_depth_map = apply_mask(equalized_img, smooth_mask)
        
        patch_bank = build_patch_bank(final_depth_map, patch_size=64, overlap_ratio=0.25)
        best_idx, best_patch, max_std_val = get_best_initial_patch(patch_bank)
        
        print("Best calculated threshold is:", optimal_thresh)
        print(f"Patch bank built! Total: {len(patch_bank)} patches.")
        print(f"Best initial patch is index {best_idx} (Std Dev = {max_std_val:.2f}).")
        
        cv2.imshow("1. Original Gray", gray_img)
        cv2.imshow("2. Otsu Mask", mask_img)
        cv2.imshow("3. FINAL CNC DEPTH MAP", final_depth_map)
        cv2.imshow("4. Best Initial Patch", best_patch)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Image not found.")