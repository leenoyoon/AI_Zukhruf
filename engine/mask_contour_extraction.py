from typing import List, Sequence, Tuple
import cv2
import numpy as np

Point = Tuple[float, float]
Path = List[Point]

try:
    from skimage import measure
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False


def _ensure_closed(path: Sequence[Point]) -> Path:
    if len(path) < 3:
        return []
    result = [(float(x), float(y)) for x, y in path]
    if result[0] != result[-1]:
        result.append(result[0])
    return result


def _contours_from_level_mask_subpixel(
    distance_mm: np.ndarray,
    level_mm: float,
    pixel_to_mm: float,
    min_path_length_mm: float,
) -> List[Path]:
    if not _HAVE_SKIMAGE:
        level_mask = np.where(distance_mm >= level_mm, 255, 0).astype(np.uint8)
        return _contours_from_level_mask(level_mask, pixel_to_mm, min_path_length_mm)

    raw_contours = measure.find_contours(distance_mm, level=level_mm)
    paths: List[Path] = []
    min_length_px = min_path_length_mm / pixel_to_mm

    for rc in raw_contours:
        if len(rc) < 3:
            continue
        pts_px = [(float(x), float(y)) for y, x in rc]  
        arr = np.array(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        perimeter_px = cv2.arcLength(arr, True)
        if perimeter_px < min_length_px:
            continue

        epsilon_mm = max(0.02, 0.6 * pixel_to_mm)
        epsilon_px = epsilon_mm / pixel_to_mm
        arr_simplified = cv2.approxPolyDP(arr, epsilon_px, True)
        pts_px = [(p[0][0], p[0][1]) for p in arr_simplified]
        path = _ensure_closed(
            [(x * pixel_to_mm, y * pixel_to_mm) for x, y in pts_px]
        )
        if len(path) >= 4:
            paths.append(path)

    return paths


def _contours_from_level_mask(
    level_mask: np.ndarray,
    pixel_to_mm: float,
    min_path_length_mm: float,
) -> List[Path]:
    contours, hierarchy = cv2.findContours(
        level_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    paths: List[Path] = []
    min_length_px = min_path_length_mm / pixel_to_mm
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue

        perimeter_px = cv2.arcLength(contour, True)
        if perimeter_px < min_length_px:
            continue

        epsilon_mm = max(0.02, 0.6 * pixel_to_mm)
        epsilon_px = epsilon_mm / pixel_to_mm
        contour = cv2.approxPolyDP(contour, epsilon_px, True)
        points = np.squeeze(contour, axis=1)
        path = _ensure_closed(
            [
                (float(point[0]) * pixel_to_mm,
                 float(point[1]) * pixel_to_mm)
                for point in points
            ]
        )
        if len(path) >= 4:
            paths.append(path)

    return paths