from typing import List, Tuple
import math
from contour_pipeline import Contour, simplify_pipeline, simplify_pipeline_by_tags
Point = Tuple[float, float]

def simplify_offset_paths(offset_paths: List[List[Point]], epsilon_mm: float=0.15, pixels_per_mm: float=1.0, min_segment_mm: float=0.02, verbose: bool=True) -> List[List[Point]]:
    contours = []
    for idx, ring in enumerate(offset_paths):
        pts = list(ring)
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        contours.append(Contour(points=pts, closed=True, is_hole=False, contour_id=idx))
    simplified, reports = simplify_pipeline(contours, epsilon_mm=epsilon_mm, pixels_per_mm=pixels_per_mm, min_segment_mm=min_segment_mm)
    out: List[List[Point]] = []
    total_before = total_after = 0
    for c, r in zip(simplified, reports):
        pts = list(c.points)
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        out.append(pts)
        total_before += r.input_points
        total_after += r.output_points
    if verbose:
        pct = 100.0 * (1 - total_after / total_before) if total_before else 0.0
        print(f'[dphull] {len(offset_paths)} ring(s): {total_before} -> {total_after} pts ({pct:.1f}% reduction)')
    return out

def _turn_angle_deg(p_prev: Point, p: Point, p_next: Point) -> float:
    v1 = (p[0] - p_prev[0], p[1] - p_prev[1])
    v2 = (p_next[0] - p[0], p_next[1] - p[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-09 or n2 < 1e-09:
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
        elif nxt < 0 or nxt >= n:
            return points[nxt - step]
        acc += math.hypot(points[nxt][0] - points[idx][0], points[nxt][1] - points[idx][1])
        idx = nxt
        if acc >= min_dist:
            return points[idx]
    return points[idx]
import numpy as np

def classify_points_straight_curve_corner(pts_mm, closed=True, lookback_mm=1.7, straight_angle_deg=8.0, corner_angle_deg=35.0):
    pts = np.asarray(pts_mm, dtype=float)
    n = len(pts)
    if n < 5:
        return ['straight'] * n

    def walk(idx, dist, direction):
        remain = dist
        i = idx
        while remain > 0:
            j = (i + direction) % n if closed else i + direction
            if not closed:
                if j < 0:
                    return 0
                if j >= n:
                    return n - 1
            d = np.linalg.norm(pts[j] - pts[i])
            remain -= d
            i = j
        return i
    tags = []
    for i in range(n):
        a = walk(i, lookback_mm, -1)
        b = walk(i, lookback_mm, +1)
        if closed:
            idxs = []
            k = a
            while True:
                idxs.append(k)
                if k == b:
                    break
                k = (k + 1) % n
        else:
            lo = min(a, b)
            hi = max(a, b)
            idxs = list(range(lo, hi + 1))
        neighborhood = pts[idxs]
        bbox = np.linalg.norm(neighborhood[-1] - neighborhood[0])
        LINE_TOL = np.clip(bbox * 0.006, 0.02, 0.05)
        p0 = neighborhood.mean(axis=0)
        uu, ss, vh = np.linalg.svd(neighborhood - p0)
        direction = vh[0]
        if len(ss) >= 2:
            linearity = ss[0] / (ss[1] + 1e-09)
        else:
            linearity = 9999.0
        diff = neighborhood - p0
        proj = diff @ direction
        closest = np.outer(proj, direction)
        perp = diff - closest
        max_dev = np.max(np.linalg.norm(perp, axis=1))
        prev = pts[a]
        curr = pts[i]
        nxt = pts[b]
        v1 = curr - prev
        v2 = nxt - curr
        if np.linalg.norm(v1) < 1e-09 or np.linalg.norm(v2) < 1e-09:
            tags.append('straight')
            continue
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
        if linearity > 18 and max_dev < LINE_TOL * 1.5:
            tags.append('straight')
        elif max_dev < LINE_TOL and ang < straight_angle_deg:
            tags.append('straight')
        elif ang > corner_angle_deg:
            tags.append('corner')
        else:
            tags.append('curve')
    return tags

def detect_corners_by_lookback(pts_mm, closed=True, lookback_mm=1.7, corner_angle_deg=35.0):
    pts = np.asarray(pts_mm, dtype=float)
    n = len(pts)
    if n < 5:
        return [0]

    def walk(idx, dist, direction):
        remain = dist
        i = idx
        for _ in range(n):
            j = (i + direction) % n if closed else i + direction
            if not closed:
                if j < 0:
                    return 0
                if j >= n:
                    return n - 1
            d = np.linalg.norm(pts[j] - pts[i])
            remain -= d
            i = j
            if remain <= 0:
                break
        return i
    raw_corners = []
    for i in range(n):
        a = walk(i, lookback_mm, -1)
        b = walk(i, lookback_mm, +1)
        v1 = pts[i] - pts[a]
        v2 = pts[b] - pts[i]
        n1, n2 = (np.linalg.norm(v1), np.linalg.norm(v2))
        if n1 < 1e-09 or n2 < 1e-09:
            continue
        v1, v2 = (v1 / n1, v2 / n2)
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
        if ang > corner_angle_deg:
            raw_corners.append(i)
    if not raw_corners:
        return [0]
    raw_corners = sorted(set(raw_corners))
    cluster_gap = max(2, int(round(lookback_mm / 2)))
    merged = [raw_corners[0]]
    for c in raw_corners[1:]:
        if c - merged[-1] <= cluster_gap:
            continue
        merged.append(c)
    if closed and len(merged) > 1 and (n - merged[-1] + merged[0] <= cluster_gap):
        merged.pop()
    return merged

def classify_points_by_segments(pts_mm, closed=True, lookback_mm=1.7, corner_angle_deg=35.0, straight_linearity_min=15.0, straight_dev_tol_mm=None):
    pts = np.asarray(pts_mm, dtype=float)
    n = len(pts)
    if n < 5:
        return ['straight'] * n
    corners = detect_corners_by_lookback(pts_mm, closed=closed, lookback_mm=lookback_mm, corner_angle_deg=corner_angle_deg)
    corners = sorted(set(corners))
    if len(corners) < 2:
        segments = [(corners[0], corners[0] + n)] if closed else [(0, n - 1)]
    else:
        segments = []
        for k in range(len(corners)):
            s = corners[k]
            e = corners[(k + 1) % len(corners)]
            if k == len(corners) - 1:
                if closed:
                    e += n
                else:
                    e = n - 1
            segments.append((s, e))
    tags = ['curve'] * n
    for s, e in segments:
        idxs = [i % n for i in range(s, e + 1)]
        seg_pts = pts[idxs]
        if len(seg_pts) <= 2:
            seg_tag = 'straight'
        elif len(seg_pts) < 4:
            seg_tag = 'curve'
        else:
            p0 = seg_pts.mean(axis=0)
            uu, ss, vh = np.linalg.svd(seg_pts - p0)
            direction = vh[0]
            linearity = ss[0] / (ss[1] + 1e-09) if len(ss) >= 2 else 9999.0
            diff = seg_pts - p0
            proj = diff @ direction
            perp = diff - np.outer(proj, direction)
            max_dev = np.max(np.linalg.norm(perp, axis=1))
            bbox = np.linalg.norm(seg_pts[-1] - seg_pts[0])
            tol = straight_dev_tol_mm if straight_dev_tol_mm is not None else np.clip(bbox * 0.006, 0.02, 0.05)
            seg_tag = 'straight' if linearity > straight_linearity_min and max_dev < tol * 1.5 else 'curve'
        for i in idxs:
            tags[i] = seg_tag
    for c in corners:
        tags[c % n] = 'corner'
    return tags

def smooth_point_tags(tags):
    if len(tags) < 3:
        return tags
    tags = list(tags)
    changed = True
    while changed:
        changed = False
        new = tags[:]
        for i in range(1, len(tags) - 1):
            if tags[i - 1] == tags[i + 1] and tags[i] != tags[i - 1]:
                new[i] = tags[i - 1]
                changed = True
        tags = new
    return tags
import numpy as np

def regularize_straight_runs(points, tags):
    pts = np.asarray(points, dtype=float).copy()
    runs = []
    start = 0
    for i in range(1, len(tags)):
        if tags[i] != tags[start]:
            runs.append((start, i - 1, tags[start]))
            start = i
    runs.append((start, len(tags) - 1, tags[start]))
    for s, e, tag in runs:
        if tag != 'straight':
            continue
        if e - s < 3:
            continue
        run = pts[s:e + 1]
        center = run.mean(axis=0)
        uu, ss, vh = np.linalg.svd(run - center)
        direction = vh[0]
        proj = (run - center) @ direction
        projected = center + np.outer(proj, direction)
        pts[s:e + 1] = projected
    return [tuple(p) for p in pts]

def simplify_offset_paths_with_curve_tags(offset_paths: List[List[Point]], epsilon_mm: float=0.15, pixels_per_mm: float=1.0, min_segment_mm: float=0.02, lookback_mm: float=1.7, straight_angle_deg: float=5.0, corner_angle_deg: float=35.0, classifier: str='segment', verbose: bool=True) -> Tuple[List[List[Point]], List[List[str]]]:
    contours = []
    for idx, ring in enumerate(offset_paths):
        pts = list(ring)
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if classifier == 'segment':
            point_tags = classify_points_by_segments(pts, closed=True, lookback_mm=lookback_mm, corner_angle_deg=corner_angle_deg)
        elif classifier == 'pointwise':
            point_tags = classify_points_straight_curve_corner(pts, closed=True, lookback_mm=lookback_mm, straight_angle_deg=straight_angle_deg, corner_angle_deg=corner_angle_deg)
        else:
            raise ValueError(f"unknown classifier={classifier!r}, expected 'segment' or 'pointwise'")
        point_tags = smooth_point_tags(point_tags)
        pts = regularize_straight_runs(pts, point_tags)
        contours.append(Contour(points=pts, closed=True, is_hole=False, contour_id=idx, metadata=point_tags))
    simplified, reports = simplify_pipeline_by_tags(contours, curve_epsilon_mm=epsilon_mm, pixels_per_mm=pixels_per_mm, min_segment_mm=min_segment_mm)
    out_paths: List[List[Point]] = []
    out_tags: List[List[str]] = []
    total_before = total_after = 0
    for c, r in zip(simplified, reports):
        pts = list(c.points)
        tags = list(c.metadata) if c.metadata is not None else ['curve'] * len(pts)
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
            tags.append(tags[0] if tags else 'curve')
        out_paths.append(pts)
        out_tags.append(tags)
        total_before += r.input_points
        total_after += r.output_points
    if verbose:
        pct = 100.0 * (1 - total_after / total_before) if total_before else 0.0
        print(f'[dphull] {len(offset_paths)} ring(s): {total_before} -> {total_after} pts ({pct:.1f}% reduction)')
    return (out_paths, out_tags)
