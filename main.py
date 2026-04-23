import cv2
import numpy as np

from core.patch_bank import build_patch_bank, get_best_initial_patch
from core.ssd_matching import generate_texture_canvas

def create_standalone_canvas(image, canvas_width, canvas_height):
    h, w = image.shape[:2]
    scale = min(canvas_width / w, canvas_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if scale > 1.0:
        blurred = cv2.GaussianBlur(resized_img, (5, 5), 0)
        _, resized_img = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas.fill(255) 
    
    x_offset = (canvas_width - new_w) // 2
    y_offset = (canvas_height - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return canvas

if __name__ == "__main__":
    image_path = r"data\input_images\pattern16.jpg"  
    USER_SELECTED_MODE = "PATTERN"  
    target_canvas_w = 800
    target_canvas_h = 800
    original_img = cv2.imread(image_path)
    if original_img is not None:
        print(f"\n--- Processing in [{USER_SELECTED_MODE}] Mode ---")
        original_img = cv2.resize(original_img, (512, 512))
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        if USER_SELECTED_MODE == "PATTERN":
            
            equalized_img = cv2.equalizeHist(gray_img)
            _, mask_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            smooth_mask = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
            
            final_depth_map = cv2.bitwise_and(equalized_img, equalized_img, mask=smooth_mask)

        elif USER_SELECTED_MODE == "LOGO":
            
            kernel_sharpening = np.array([[-1, -1, -1], 
                                          [-1,  9, -1],
                                          [-1, -1, -1]])
            sharpened = cv2.filter2D(gray_img, -1, kernel_sharpening)
            _, final_depth_map = cv2.threshold(sharpened, 120, 255, cv2.THRESH_BINARY)
            final_depth_map = cv2.morphologyEx(final_depth_map, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        elif USER_SELECTED_MODE == "ORNAMENT":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_img)
            _, final_depth_map = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if USER_SELECTED_MODE == "PATTERN":
            patch_size = 128
            overlap_width = patch_size // 6
            num_patches_x = target_canvas_w // (patch_size - overlap_width)
            num_patches_y = target_canvas_h // (patch_size - overlap_width)

            patch_bank = build_patch_bank(final_depth_map, patch_size=patch_size, overlap_ratio=0.4)
            best_idx, best_patch, max_std_val = get_best_initial_patch(patch_bank)
    
            final_canvas = generate_texture_canvas(patch_bank, best_patch, num_patches_y, num_patches_x, patch_size, overlap_width)
        else:
            final_canvas = create_standalone_canvas(final_depth_map, target_canvas_w, target_canvas_h)
            
        cv2.imshow("1. Original", original_img)
        cv2.imshow("2. Processed Depth Map", final_depth_map)
        cv2.imshow("3. FINAL CNC CANVAS", final_canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Image not found.")