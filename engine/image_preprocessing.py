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
    border_sample_width_px: int = 5,   
) -> Tuple[np.ndarray, int]:
    pad_px = int(np.ceil(tool_dia_px / 2.0 * (1.0 + extra_margin_ratio)))
    pad_px = max(pad_px, 2)
    if border_value is None:  
        h, w = img.shape[:2]
        s = min(border_sample_width_px, h // 4, w // 4, 1) if min(h, w) > 4 else 1
        strips = [
            img[0:s, :],       
            img[-s:, :],       
            img[:, 0:s],       
            img[:, -s:],       
        ]
        border_value = int(np.median(np.concatenate([st.reshape(-1) for st in strips])))
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
    k = int(round((tool_dia_mm / pixel_to_mm) * 0.5))
    k = max(3, k)
    if k % 2 == 0:
        k += 1  
    return k

def estimate_adaptive_upscale_factor(
    image_bgr: np.ndarray,
    wood_width_mm: float,
    wood_height_mm: float,
    tool_dia_mm: float,
    fit_mode: Literal["contain", "cover"] = "contain",
    min_factor: int = 2,
    max_factor: int = 6,
) -> Tuple[int, List[str]]:
    notes: List[str] = []
    base_pixel_to_mm, _ = compute_pixel_to_mm(
        image_bgr.shape[:2], wood_width_mm, wood_height_mm, tool_dia_mm, fit_mode
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    corners_mean = np.mean([gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]])
    thresh_type = (
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
        if corners_mean < 127 else
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary = cv2.threshold(blurred, 0, 255, thresh_type)
    corner_pts = cv2.goodFeaturesToTrack(
        binary, maxCorners=500, qualityLevel=0.05, minDistance=3,
    )
    if corner_pts is None or len(corner_pts) < 4:
        notes.append(
            f"زوايا حادة قليلة جداً -- الشكل بسيط، upscale_factor={min_factor} كافٍ."
        )
        return min_factor, notes
    pts_mm = corner_pts.reshape(-1, 2) * base_pixel_to_mm
    diff = pts_mm[:, None, :] - pts_mm[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_dists = dist_matrix.min(axis=1)
    nearest_dists = nearest_dists[np.isfinite(nearest_dists)]
    if len(nearest_dists) == 0:
        return min_factor, notes

    tightness_mm = float(np.percentile(nearest_dists, 10))
    ratio = tightness_mm / tool_dia_mm
    if ratio >= 4.0:
        factor = min_factor
    elif ratio >= 2.0:
        factor = min_factor + 1
    elif ratio >= 1.0:
        factor = min_factor + 2
    else:
        factor = max_factor

    factor = int(np.clip(factor, min_factor, max_factor))
    notes.append(
        f"كثافة الزوايا: أقرب تباعد بين زوايا متقاربة (p10) ≈ {tightness_mm:.3f} mm "
        f"(نسبة لقطر الأداة {tool_dia_mm}mm = {ratio:.2f}x) -> "
        f"upscale_factor تلقائي = {factor}"
    )
    return factor, notes


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

def adaptive_blur_kernel_size(tool_dia_mm: float, pixel_to_mm: float) -> int:
    k = max(3, int(round((tool_dia_mm / pixel_to_mm) * 0.15)))
    if k % 2 == 0:
        k += 1
    return k

def binarize(
    gray: np.ndarray,
    method: Literal["otsu", "adaptive"] = "otsu",
    invert_if_dark_bg: bool = True,
    blur_kernel_size: int = 3,
    use_bilateral: bool = True,   
) -> np.ndarray:
    if use_bilateral:
        blurred = cv2.bilateralFilter(
            gray, d=blur_kernel_size, sigmaColor=50, sigmaSpace=50
        )
    else:
        blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

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


def detect_uneven_lighting(gray: np.ndarray, tile_grid: int = 4) -> bool:
    h, w = gray.shape[:2]
    tile_h, tile_w = h // tile_grid, w // tile_grid
    means = []
    for r in range(tile_grid):
        for c in range(tile_grid):
            tile = gray[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
            if tile.size > 0:
                means.append(float(np.mean(tile)))
    if len(means) < 2:
        return False
    spread = (max(means) - min(means)) / 255.0
    return spread > 0.25  


def remove_small_islands(
    binary: np.ndarray,
    pixel_to_mm: float,
    min_island_area_mm2: float = 0.15,
) -> Tuple[np.ndarray, int]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    pixel_area_mm2 = pixel_to_mm * pixel_to_mm
    min_area_px = min_island_area_mm2 / pixel_area_mm2

    cleaned = np.zeros_like(binary)
    removed_count = 0
    for label_id in range(1, num_labels):  
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px >= min_area_px:
            cleaned[labels == label_id] = 255
        else:
            removed_count += 1

    return cleaned, removed_count

def preprocess_pipeline(
    image_bgr: np.ndarray,
    wood_width_mm: float,
    wood_height_mm: float,
    tool_dia_mm: float,
    threshold_method: Optional[Literal["otsu", "adaptive"]] = None,   
    use_clahe: bool = False,
    fit_mode: Literal["contain", "cover"] = "contain",
    upscale_factor: Optional[int] = None,
    use_bilateral: bool = True,
) -> PreprocessResult:
    notes: List[str] = []
    if upscale_factor is None:
        upscale_factor, upscale_notes = estimate_adaptive_upscale_factor(
            image_bgr, wood_width_mm, wood_height_mm, tool_dia_mm, fit_mode
        )
        notes.extend(upscale_notes)
    
    MAX_DIM_PX = 8000
    h0, w0 = image_bgr.shape[:2]
    max_current_dim = max(h0, w0)
    if upscale_factor > 1 and max_current_dim * upscale_factor > MAX_DIM_PX:
        capped_factor = max(1, int(MAX_DIM_PX / max_current_dim))
        notes.append(
            f"تحذير: upscale_factor خُفّض من {upscale_factor} لـ{capped_factor} "
            f"لأنو الصورة الأصلية ({w0}x{h0}) كبيرة والناتج كان رح يتجاوز "
            f"{MAX_DIM_PX}px (بطء/استهلاك ذاكرة غير ضروري)."
        )
        upscale_factor = capped_factor

    if upscale_factor > 1:
        image_bgr = cv2.resize(
            image_bgr, None,
            fx=upscale_factor, fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
    pixel_to_mm, scale_notes = compute_pixel_to_mm(
        image_bgr.shape[:2], wood_width_mm, wood_height_mm, tool_dia_mm, fit_mode
    )
    notes.extend(scale_notes)
    tool_dia_px = tool_dia_mm / pixel_to_mm
    padded_bgr, pad_px = pad_for_border_touching_shapes(image_bgr, tool_dia_px)
    gray = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        gray = enhance_contrast_clahe(gray)

    if threshold_method is None:
        threshold_method = "adaptive" if detect_uneven_lighting(gray) else "otsu"
        notes.append(f"threshold method تلقائي = {threshold_method}")

    blur_kernel_size = adaptive_blur_kernel_size(tool_dia_mm, pixel_to_mm)
    binary = binarize(
        gray,
        method=threshold_method,
        blur_kernel_size=blur_kernel_size,
        use_bilateral=use_bilateral,
    )
    kernel_size = adaptive_morph_kernel_size(tool_dia_mm, pixel_to_mm)
    binary = clean_binary(binary, kernel_size)  
    binary, removed_islands = remove_small_islands(binary, pixel_to_mm)
    if removed_islands:
        notes.append(
            f"تم حذف {removed_islands} منطقة/جزيرة صغيرة (< 0.15 mm²) "
            f"محتمل تكون ضجيج وليست تفاصيل تصميم."
        )
    foreground_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    if foreground_ratio < 0.005:
        notes.append(
            f"تحذير: نسبة البكسلات البيضاء منخفضة جداً ({foreground_ratio:.4%}) -- "
            f"تأكدي إنو threshold صحيح أو إنو الصورة فعلاً فيها تفاصيل كافية."
        )
    elif foreground_ratio > 0.95:
        notes.append(
            f"تحذير: نسبة البكسلات البيضاء عالية جداً ({foreground_ratio:.4%}) -- "
            f"محتمل الـ threshold انعكس أو الصورة كلها تقريباً foreground."
        )
    return PreprocessResult(
        binary=binary,
        pixel_to_mm=pixel_to_mm,
        pad_px=pad_px,
        scale_notes=notes,
    )