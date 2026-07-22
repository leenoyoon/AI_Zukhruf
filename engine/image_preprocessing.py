from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
import numpy as np
import cv2

@dataclass
class PreprocessResult:
    binary: np.ndarray            
    pixel_to_mm: float             
    pad_px: int                    
    scale_notes: List[str]         


def merge_composite_layers(
    layers: List[np.ndarray],
    mode: Literal["union", "priority"] = "union",
) -> np.ndarray:

    if not layers:
        raise ValueError("لازم طبقة وحدة عالأقل")
    shape = layers[0].shape
    for L in layers:
        if L.shape != shape:
            raise ValueError(
                "الطبقات لازم تكون بنفس الأبعاد تماماً قبل الدمج — "
                "لو أبعادها مختلفة، هيدا خطأ يلي لازم ينحل بمرحلة الـ mapping "
                "(انظر compute_pixel_to_mm) قبل ما توصل لهون."
            )

    if mode == "union":
        merged = np.zeros(shape, dtype=np.uint8)
        for L in layers:
            merged = cv2.bitwise_or(merged, L)
        return merged

    
    merged = layers[0].copy()
    for L in layers[1:]:
        
        empty_mask = merged == 0
        merged[empty_mask] = L[empty_mask]
    return merged


def pad_for_border_touching_shapes(
    img: np.ndarray,
    tool_dia_px: float,
    extra_margin_ratio: float = 0.5,
    border_value: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    pad_px = int(np.ceil(tool_dia_px / 2.0 * (1.0 + extra_margin_ratio)))
    pad_px = max(pad_px, 2)  

    if border_value is None:    
        h, w = img.shape[:2]
        corners = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
        border_value = int(np.mean(corners))

    if img.ndim == 2:
        padded = cv2.copyMakeBorder(
            img, pad_px, pad_px, pad_px, pad_px,
            borderType=cv2.BORDER_CONSTANT, value=border_value,
        )
    else:
        padded = cv2.copyMakeBorder(
            img, pad_px, pad_px, pad_px, pad_px,
            borderType=cv2.BORDER_CONSTANT, value=(border_value,) * img.shape[2],
        )
    return padded, pad_px


def compute_pixel_to_mm(
    image_shape: Tuple[int, int],
    wood_width_mm: float,
    wood_height_mm: float,
    tool_dia_mm: float,
    fit_mode: Literal["contain", "cover"] = "contain",
) -> Tuple[float, List[str]]:
    notes: List[str] = []
    img_h, img_w = image_shape[:2]

    scale_x = wood_width_mm / img_w
    scale_y = wood_height_mm / img_h

    pixel_to_mm = min(scale_x, scale_y) if fit_mode == "contain" else max(scale_x, scale_y)

    if abs(scale_x - scale_y) / max(scale_x, scale_y) > 0.02:
        notes.append(
            f"نسبة أبعاد الصورة ({img_w}x{img_h}) لا تطابق نسبة أبعاد الخشب "
            f"({wood_width_mm}x{wood_height_mm}mm) -- تم اعتماد {fit_mode} "
            f"وسيبقى هامش فاضٍ أو جزء غير مُستخدم من القطعة حسب الاتجاه."
        )

    min_feature_px = tool_dia_mm / pixel_to_mm
    notes.append(
        f"أصغر تفصيل ذو معنى (تقريباً) = {min_feature_px:.1f} بكسل "
        f"(= قطر الأداة {tool_dia_mm}mm بمقياس الصورة الحالي). "
        f"أي تفصيل أصغر من هيك لازم يُدمج/يُزال بمرحلة التنظيف المورفولوجي."
    )
    return pixel_to_mm, notes


def adaptive_morph_kernel_size(tool_dia_mm: float, pixel_to_mm: float) -> int:
    """
    حجم الـ morphological kernel متناسب مع الـ scale الفعلي (ملاحظة 
    بدل الرقم الثابت (3,3) الموجود بالكود الحالي، حتى ينفع مع صور بمقاييس مختلفة.
    """
    k = int(round((tool_dia_mm / pixel_to_mm) * 0.5))
    k = max(3, k)
    if k % 2 == 0:
        k += 1  
    return k


def stitch_patterns_with_feather(
    tiles: List[np.ndarray],
    layout: Tuple[int, int],
    feather_px: int = 4,
) -> np.ndarray:
    rows, cols = layout
    if len(tiles) != rows * cols:
        raise ValueError("عدد الـ tiles لازم يطابق rows*cols")

    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.zeros((tile_h * rows, tile_w * cols), dtype=np.float32)
    weight = np.zeros_like(canvas)
    base_w = np.ones((tile_h, tile_w), dtype=np.float32)
    if feather_px > 0:
        ramp = np.linspace(0, 1, feather_px, dtype=np.float32)
        base_w[:feather_px, :] *= ramp[:, None]
        base_w[-feather_px:, :] *= ramp[::-1, None]
        base_w[:, :feather_px] *= ramp[None, :]
        base_w[:, -feather_px:] *= ramp[None, ::-1]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            tile = tiles[idx].astype(np.float32)
            y0, x0 = r * tile_h, c * tile_w
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] += tile * base_w
            weight[y0:y0 + tile_h, x0:x0 + tile_w] += base_w
            idx += 1

    weight[weight == 0] = 1.0
    result = (canvas / weight).astype(np.uint8)
    return result

def enhance_contrast_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def binarize(
    gray: np.ndarray,
    method: Literal["otsu", "adaptive"] = "otsu",
    invert_if_dark_bg: bool = True,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if method == "otsu":
        h, w = gray.shape[:2]
        corners_mean = np.mean([gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]])
        if invert_if_dark_bg and corners_mean < 127:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    block_size = max(15, (min(gray.shape[:2]) // 8) | 1)  
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, block_size, C=5,
    )
    return binary


def clean_binary(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def preprocess_pipeline(
    image_bgr: np.ndarray,
    wood_width_mm: float,
    wood_height_mm: float,
    tool_dia_mm: float,
    threshold_method: Literal["otsu", "adaptive"] = "otsu",
    use_clahe: bool = False,
    fit_mode: Literal["contain", "cover"] = "contain",
) -> PreprocessResult:
    notes: List[str] = []
    pixel_to_mm, scale_notes = compute_pixel_to_mm(
        image_bgr.shape[:2], wood_width_mm, wood_height_mm, tool_dia_mm, fit_mode
    )
    notes.extend(scale_notes)
    tool_dia_px = tool_dia_mm / pixel_to_mm
    padded_bgr, pad_px = pad_for_border_touching_shapes(image_bgr, tool_dia_px)
    gray = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        gray = enhance_contrast_clahe(gray)

    binary = binarize(gray, method=threshold_method)
    kernel_size = adaptive_morph_kernel_size(tool_dia_mm, pixel_to_mm)
    binary = clean_binary(binary, kernel_size)
    return PreprocessResult(
        binary=binary,
        pixel_to_mm=pixel_to_mm,
        pad_px=pad_px,
        scale_notes=notes,
    )
