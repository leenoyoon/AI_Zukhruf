from typing import List, Tuple
import math
from contour_pipeline import Contour, simplify_pipeline

Point = Tuple[float, float]

def simplify_offset_paths(
    offset_paths: List[List[Point]],
    epsilon_mm: float = 0.15,
    pixels_per_mm: float = 1.0,
    min_segment_mm: float = 0.02,
    verbose: bool = True,
) -> List[List[Point]]:
    contours = []
    for idx, ring in enumerate(offset_paths):
        pts = list(ring)
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]  
        contours.append(Contour(points=pts, closed=True, is_hole=False, contour_id=idx))

    simplified, reports = simplify_pipeline(
        contours,
        epsilon_mm=epsilon_mm,
        pixels_per_mm=pixels_per_mm,
        min_segment_mm=min_segment_mm,
    )

    out: List[List[Point]] = []
    total_before = total_after = 0
    for c, r in zip(simplified, reports):
        pts = list(c.points)
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])  
        out.append(pts)
        total_before += r.input_points
        total_after += r.output_points
        if verbose and r.notes:
            print(f"[dphull] ring {r.contour_id}: {'; '.join(r.notes)}")

    if verbose:
        pct = (100.0 * (1 - total_after / total_before)) if total_before else 0.0
        print(f"[dphull] {len(offset_paths)} ring(s): {total_before} -> {total_after} "
              f"points ({pct:.1f}% reduction)")

    return out


def _turn_angle_deg(p_prev: Point, p: Point, p_next: Point) -> float:
    v1 = (p[0] - p_prev[0], p[1] - p_prev[1])
    v2 = (p_next[0] - p[0], p_next[1] - p[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def _walk_by_distance(points: List[Point], i: int, step: int, min_dist: float, closed: bool) -> Point:
    n = len(points)
    x0, y0 = points[i]
    idx = i
    acc = 0.0
    for _ in range(n):  
        nxt = idx + step
        if closed:
            nxt %= n
        else:
            if nxt < 0 or nxt >= n:
                return points[nxt - step]  
        acc += math.hypot(points[nxt][0] - points[idx][0], points[nxt][1] - points[idx][1])
        idx = nxt
        if acc >= min_dist:
            return points[idx]
    return points[idx]


def classify_points_straight_curve_corner(
    points: List[Point],
    closed: bool,
    lookback_mm: float = 1.0,
    straight_angle_deg: float = 5.0,
    corner_angle_deg: float = 35.0,
) -> List[str]:
    n = len(points)
    if n < 5:
        return ["curve"] * n

    tags: List[str] = []
    for i in range(n):
        p_prev = _walk_by_distance(points, i, -1, lookback_mm, closed)
        p_next = _walk_by_distance(points, i, +1, lookback_mm, closed)
        avg_angle = _turn_angle_deg(p_prev, points[i], p_next)

        if avg_angle > corner_angle_deg:
            tags.append("corner")
        elif avg_angle < straight_angle_deg:
            tags.append("straight")
        else:
            tags.append("curve")

    return tags


def simplify_offset_paths_with_curve_tags(
    offset_paths: List[List[Point]],
    epsilon_mm: float = 0.15,
    pixels_per_mm: float = 1.0,
    min_segment_mm: float = 0.02,
    lookback_mm: float = 1.0,
    straight_angle_deg: float = 5.0,
    corner_angle_deg: float = 35.0,
    verbose: bool = True,
) -> Tuple[List[List[Point]], List[List[str]]]:
    contours = []
    for idx, ring in enumerate(offset_paths):
        pts = list(ring)
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]

        point_tags = classify_points_straight_curve_corner(
            pts, closed=True, lookback_mm=lookback_mm,
            straight_angle_deg=straight_angle_deg,
            corner_angle_deg=corner_angle_deg,
        )
        contours.append(Contour(points=pts, closed=True, is_hole=False,
                                 contour_id=idx, metadata=point_tags))

    simplified, reports = simplify_pipeline(
        contours, epsilon_mm=epsilon_mm, pixels_per_mm=pixels_per_mm,
        min_segment_mm=min_segment_mm,
    )

    out_paths: List[List[Point]] = []
    out_tags: List[List[str]] = []
    total_before = total_after = 0
    for c, r in zip(simplified, reports):
        pts = list(c.points)
        tags = list(c.metadata) if c.metadata is not None else ["curve"] * len(pts)
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
            tags.append(tags[0] if tags else "curve")
        out_paths.append(pts)
        out_tags.append(tags)
        total_before += r.input_points
        total_after += r.output_points
        if verbose and r.notes:
            print(f"[dphull] ring {r.contour_id}: {'; '.join(r.notes)}")

    if verbose:
        pct = (100.0 * (1 - total_after / total_before)) if total_before else 0.0
        print(f"[dphull] {len(offset_paths)} ring(s): {total_before} -> {total_after} "
              f"points ({pct:.1f}% reduction), curve tags preserved")

    return out_paths, out_tags