from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
import numpy as np
import cv2


Point = Tuple[float, float]


@dataclass
class Contour:
    points: List[Point]
    closed: bool = True
    is_hole: bool = False
    contour_id: Any = None
    metadata: Optional[List[Any]] = None   


@dataclass
class ExtractionReport:
    total_found: int
    kept: int
    dropped_too_small: int
    dropped_as_background: int
    dropped_as_tile_seam: int
    orphan_parent_count: int   
    notes: List[str] = field(default_factory=list)


def adaptive_min_area_px(tool_dia_mm: float, pixel_to_mm: float, factor: float = 0.25) -> float:
    tool_radius_px = (tool_dia_mm / pixel_to_mm) / 2.0
    return factor * np.pi * (tool_radius_px ** 2)

def _is_background_frame(cnt: np.ndarray, img_shape: Tuple[int, int],
                          area: float, bbox_cover_ratio: float = 0.98,
                          fill_ratio_threshold: float = 0.98) -> bool:

    x, y, w, h = cv2.boundingRect(cnt)
    img_h, img_w = img_shape[:2]

    covers_whole_image = (w >= img_w * bbox_cover_ratio) and (h >= img_h * bbox_cover_ratio)
    if not covers_whole_image:
        return False

    bbox_area = w * h
    fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
    return fill_ratio >= fill_ratio_threshold

def extract_raw_contours(
    binary: np.ndarray,
    min_area_px: float = 4.0,
    bbox_cover_ratio: float = 0.98,
    fill_ratio_threshold: float = 0.98,
) -> Tuple[List[Contour], ExtractionReport]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    notes: List[str] = []
    kept_contours: List[Contour] = []
    dropped_small = 0
    dropped_bg = 0
    orphan_parent_count = 0

    if hierarchy is None:
        notes.append("لا يوجد أي contour بالصورة -- تأكد من صحة الـ threshold قبل هالمرحلة.")
        return [], ExtractionReport(0, 0, 0, 0, 0, 0, notes)

    hierarchy = hierarchy[0]

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        if _is_background_frame(cnt, binary.shape, area, bbox_cover_ratio, fill_ratio_threshold):
            dropped_bg += 1
            notes.append(f"contour #{i} إطار خلفية وهمي (مستطيل شبه مصمت يغطي الصورة كاملة) -- تم تجاهله.")
            continue

        if area < min_area_px:
            dropped_small += 1
            continue

        cnt_2d = np.squeeze(cnt)
        if cnt_2d.ndim == 1 or len(cnt_2d) < 3:
            dropped_small += 1
            continue

        parent_idx = hierarchy[i][3]
        is_hole = parent_idx != -1
        if not is_hole:
            orphan_parent_count += 1  

        points: List[Point] = [(float(p[0]), float(p[1])) for p in cnt_2d]
        kept_contours.append(
            Contour(points=points, closed=True, is_hole=is_hole, contour_id=i)
        )

    report = ExtractionReport(
        total_found=len(contours),
        kept=len(kept_contours),
        dropped_too_small=dropped_small,
        dropped_as_background=dropped_bg,
        dropped_as_tile_seam=0, 
        orphan_parent_count=orphan_parent_count,
        notes=notes,
    )
    return kept_contours, report

def remove_tile_seam_artifacts(
    contours: List[Contour],
    image_shape: Tuple[int, int],
    edge_tolerance_px: float = 1.5,
) -> Tuple[List[Contour], int]:
    h, w = image_shape[:2]
    kept: List[Contour] = []
    dropped = 0

    for c in contours:
        pts = np.array(c.points)
        touches_top = np.all(pts[:, 1] <= edge_tolerance_px)
        touches_bottom = np.all(pts[:, 1] >= h - 1 - edge_tolerance_px)
        touches_left = np.all(pts[:, 0] <= edge_tolerance_px)
        touches_right = np.all(pts[:, 0] >= w - 1 - edge_tolerance_px)

        if touches_top or touches_bottom or touches_left or touches_right:
            dropped += 1
            continue
        kept.append(c)

    return kept, dropped


def _turn_angle(p_prev: Point, p: Point, p_next: Point) -> float:
    """زاوية الانعطاف عند نقطة (بالراديان) -- 0 يعني استمرار خط مستقيم."""
    v1 = np.array(p) - np.array(p_prev)
    v2 = np.array(p_next) - np.array(p)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_a))


def classify_corners_vs_curves(
    contour: Contour,
    window: int = 3,
    corner_angle_threshold_deg: float = 35.0,
) -> Contour:
    n = len(contour.points)
    if n < 2 * window + 1:
        meta = ["curve"] * n
        return Contour(contour.points, contour.closed, contour.is_hole, contour.contour_id, meta)

    meta: List[str] = []
    for i in range(n):
        angles = []
        for w_ in range(1, window + 1):
            p_prev = contour.points[(i - w_) % n]
            p = contour.points[i]
            p_next = contour.points[(i + w_) % n]
            angles.append(_turn_angle(p_prev, p, p_next))
        avg_angle_deg = np.degrees(np.mean(angles))
        meta.append("corner" if avg_angle_deg > corner_angle_threshold_deg else "curve")

    return Contour(contour.points, contour.closed, contour.is_hole, contour.contour_id, meta)


def contour_extraction_pipeline(
    binary: np.ndarray,
    tool_dia_mm: Optional[float] = None,
    pixel_to_mm: Optional[float] = None,
    min_area_px: Optional[float] = None,
    classify_curves: bool = True,
    is_tiled_pattern: bool = False,
) -> Tuple[List[Contour], ExtractionReport]:
    notes: List[str] = []
    if min_area_px is None:
        if tool_dia_mm is not None and pixel_to_mm is not None:
            min_area_px = adaptive_min_area_px(tool_dia_mm, pixel_to_mm)
        else:
            min_area_px = 4.0
            notes.append()

    contours, report = extract_raw_contours(binary, min_area_px=min_area_px)
    report.notes = notes + report.notes

    if is_tiled_pattern:
        contours, dropped_seam = remove_tile_seam_artifacts(contours, binary.shape)
        report.dropped_as_tile_seam = dropped_seam
        report.kept = len(contours)

    if classify_curves:
        contours = [classify_corners_vs_curves(c) for c in contours]

    return contours, report