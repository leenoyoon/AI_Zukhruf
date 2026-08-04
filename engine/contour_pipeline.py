from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Sequence

from dphull import DPHull, Point, cross

@dataclass
class Contour:
    points: List[Point]
    closed: bool = True
    is_hole: bool = False
    contour_id: Any = None
    metadata: Optional[List[Any]] = None


@dataclass
class SimplifyReport:
    contour_id: Any
    input_points: int
    output_points: int
    dropped_near_duplicates: int
    had_self_intersections_before: bool
    had_self_intersections_after: bool
    fell_back_to_original: bool
    notes: List[str] = field(default_factory=list)


# 6. Near-duplicate
def dedupe_points(
    points: List[Point],
    closed: bool,
    min_dist: float,
    metadata: Optional[List[Any]] = None,
) -> Tuple[List[Point], Optional[List[Any]], int]:
    if not points:
        return points, metadata, 0

    min_dist_sq = min_dist * min_dist
    out_pts = [points[0]]
    out_meta = [metadata[0]] if metadata is not None else None
    dropped = 0

    for idx in range(1, len(points)):
        px, py = points[idx]
        lx, ly = out_pts[-1]
        if (px - lx) ** 2 + (py - ly) ** 2 < min_dist_sq:
            dropped += 1
            continue
        out_pts.append(points[idx])
        if out_meta is not None:
            out_meta.append(metadata[idx])

    if closed and len(out_pts) > 1:
        px, py = out_pts[0]
        lx, ly = out_pts[-1]
        if (px - lx) ** 2 + (py - ly) ** 2 < min_dist_sq:
            out_pts.pop()
            if out_meta is not None:
                out_meta.pop()
            dropped += 1

    return out_pts, out_meta, dropped


# 2. Self-intersection
def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12 and
            min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


try:
    from shapely.geometry import LinearRing, LineString
    _HAVE_SHAPELY = True
except ImportError:
    _HAVE_SHAPELY = False


def _has_self_intersections_bruteforce(points: List[Point], closed: bool) -> bool:
    n = len(points)
    if n < 4:
        return False

    edges = [(i, (i + 1) % n) for i in range(n - (0 if closed else 1))]
    m = len(edges)
    for a in range(m):
        i1, i2 = edges[a]
        for b in range(a + 1, m):
            j1, j2 = edges[b]
            if len({i1, i2, j1, j2}) < 4:
                continue
            if _segments_intersect(points[i1], points[i2], points[j1], points[j2]):
                return True
    return False


def has_self_intersections(points: List[Point], closed: bool) -> bool:
    n = len(points)
    if n < 4:
        return False

    if _HAVE_SHAPELY:
        try:
            geom = LinearRing(points) if closed else LineString(points)
            return not geom.is_simple
        except Exception:
            pass

    return _has_self_intersections_bruteforce(points, closed)



# 1. Closed-contour support
def _convex_hull_indices(points: List[Point]) -> List[int]:
    n = len(points)
    if n < 3:
        return list(range(n))
    order = sorted(range(n), key=lambda i: points[i])

    def turn(o: int, a: int, b: int) -> float:
        return cross(points[o], points[a], points[b])

    lower: List[int] = []
    for i in order:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    upper: List[int] = []
    for i in reversed(order):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def _farthest_pair(points: List[Point]) -> Tuple[int, int]:
    n = len(points)
    if n <= 1:
        return (0, 0)
    if n == 2:
        return (0, 1)

    hull = _convex_hull_indices(points)
    h = len(hull)
    if h < 3:
        return (hull[0], hull[-1]) if h == 2 else (0, n - 1)

    def dist_sq(i: int, j: int) -> float:
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        return dx * dx + dy * dy

    best_d = -1.0
    best = (hull[0], hull[0])
    j = 1
    for i in range(h):
        ni = (i + 1) % h
        while True:
            nj = (j + 1) % h
            area_j = abs(cross(points[hull[i]], points[hull[ni]], points[hull[j]]))
            area_nj = abs(cross(points[hull[i]], points[hull[ni]], points[hull[nj]]))
            if area_nj > area_j:
                j = nj
            else:
                break
        for pair in ((hull[i], hull[j]), (hull[ni], hull[j])):
            d = dist_sq(*pair)
            if d > best_d:
                best_d, best = d, pair
    return best


def split_closed_to_open(
    points: List[Point], i_anchor: int, j_anchor: int
) -> Tuple[List[Point], List[Point]]:
    n = len(points)
    i, j = sorted((i_anchor, j_anchor))
    chain_a = points[i:j + 1]
    chain_b = points[j:] + points[:i + 1]
    return chain_a, chain_b


def merge_open_chains(simplified_a: List[Point], simplified_b: List[Point]) -> List[Point]:
    return simplified_a[:-1] + simplified_b[:-1]


# 4/5/6/7. Full per-contour simplification
def simplify_contour(
    contour: Contour,
    epsilon_mm: float,
    pixels_per_mm: float = 1.0,
    min_segment_mm: float = 0.02,
    validate: bool = True,
) -> Tuple[Contour, SimplifyReport]:
    notes: List[str] = []
    n_in = len(contour.points)
    epsilon_px = epsilon_mm * pixels_per_mm
    min_seg_px = min_segment_mm * pixels_per_mm

    pts, meta, dropped = dedupe_points(
        contour.points, contour.closed, min_seg_px, contour.metadata
    )
    if dropped:
        notes.append(f"dropped {dropped} near-duplicate point(s) (< {min_segment_mm} mm apart)")

    if len(pts) < (3 if contour.closed else 2):
        notes.append("too few points after cleanup; returning as-is")
        report = SimplifyReport(contour.contour_id, n_in, len(pts), dropped,
                                 False, False, True, notes)
        return Contour(pts, contour.closed, contour.is_hole, contour.contour_id, meta), report

    intersects_before = has_self_intersections(pts, contour.closed) if validate else False
    if intersects_before:
        notes.append(
            "input contour is self-intersecting (likely a Path-Offsetting "
            "artifact at a sharp corner) -- simplifying anyway, but flag "
            "this contour for review; consider fixing the offset join type"
        )

    try:
        if contour.closed:
            i_anchor, j_anchor = _farthest_pair(pts)
            chain_a, chain_b = split_closed_to_open(pts, i_anchor, j_anchor)
            i, j = sorted((i_anchor, j_anchor))
            map_a = list(range(i, j + 1))
            map_b = list(range(j, len(pts))) + list(range(0, i + 1))

            idx_a = DPHull(chain_a, epsilon_px).run_indices()
            idx_b = DPHull(chain_b, epsilon_px).run_indices()

            simp_a = [chain_a[k] for k in idx_a]
            simp_b = [chain_b[k] for k in idx_b]
            out_pts = merge_open_chains(simp_a, simp_b)

            if meta is not None:
                global_a = [map_a[k] for k in idx_a][:-1]
                global_b = [map_b[k] for k in idx_b][:-1]
                out_meta = [meta[g] for g in global_a] + [meta[g] for g in global_b]
            else:
                out_meta = None
        else:
            h = DPHull(pts, epsilon_px)
            idx = h.run_indices()
            out_pts = [pts[k] for k in idx]
            out_meta = [meta[k] for k in idx] if meta is not None else None
    except Exception as exc:
        notes.append(f"simplification failed ({exc!r}); falling back to cleaned-up original")
        report = SimplifyReport(contour.contour_id, n_in, len(pts), dropped,
                                 intersects_before, intersects_before, True, notes)
        return Contour(pts, contour.closed, contour.is_hole, contour.contour_id, meta), report

    intersects_after = has_self_intersections(out_pts, contour.closed) if validate else False
    fell_back = False
    if intersects_after and not intersects_before:
        notes.append(
            "simplification introduced a self-intersection that wasn't in "
            "the input -- discarding the simplified version and keeping "
            "the cleaned-up original for this contour"
        )
        out_pts, out_meta = pts, meta
        fell_back = True

    report = SimplifyReport(
        contour_id=contour.contour_id,
        input_points=n_in,
        output_points=len(out_pts),
        dropped_near_duplicates=dropped,
        had_self_intersections_before=intersects_before,
        had_self_intersections_after=intersects_after,
        fell_back_to_original=fell_back,
        notes=notes,
    )
    return Contour(out_pts, contour.closed, contour.is_hole, contour.contour_id, out_meta), report


# 4b. Tag-aware simplification (straight segments -> collapse to 2 points,
#     curve/corner segments -> left essentially untouched so G2/G3 fitting
#     downstream still has the real point cloud to work with)
def _split_into_runs(tags: List[Any]) -> List[Tuple[int, int, Any]]:
    """Return (start_idx, end_idx_inclusive, tag) for each run of consecutive
    equal tags."""
    runs: List[Tuple[int, int, Any]] = []
    n = len(tags)
    if n == 0:
        return runs
    start = 0
    for i in range(1, n):
        if tags[i] != tags[start]:
            runs.append((start, i - 1, tags[start]))
            start = i
    runs.append((start, n - 1, tags[start]))
    return runs


def _rotate_to_run_boundary(
    points: List[Point], tags: List[Any]
) -> Tuple[List[Point], List[Any]]:
    """Rotate a closed ring so index 0 sits exactly on a tag change. This
    avoids a run wrapping around the array seam and being split in two by
    mistake. If every point has the same tag, returns the input unchanged."""
    n = len(tags)
    if n < 2:
        return points, tags
    start = None
    for i in range(n):
        if tags[i] != tags[i - 1]:  # tags[-1] wraps to the last element, as intended
            start = i
            break
    if start is None or start == 0:
        return points, tags
    return points[start:] + points[:start], tags[start:] + tags[:start]

def merge_straight_runs(points, runs,
                        max_bridge_points=2,
                        line_tol=0.05):
    """
    Merge:

        Straight
        tiny Curve
        Straight

    if both straight runs belong to the same line.
    """

    import numpy as np

    merged = []

    i = 0

    while i < len(runs):

        if i + 2 < len(runs):

            s1, e1, t1 = runs[i]
            s2, e2, t2 = runs[i+1]
            s3, e3, t3 = runs[i+2]

            if (
                t1 == "straight"
                and t2 == "curve"
                and t3 == "straight"
                and (e2 - s2 + 1) <= max_bridge_points
            ):

                p0 = np.array(points[s1])
                p1 = np.array(points[e3])

                d = p1 - p0

                L = np.linalg.norm(d)

                if L > 1e-9:

                    d /= L

                    ok = True

                    for k in range(s1, e3 + 1):

                        p = np.array(points[k])

                        proj = p0 + np.dot(p - p0, d) * d

                        err = np.linalg.norm(p - proj)

                        if err > line_tol:

                            ok = False

                            break

                    if ok:

                        merged.append((s1, e3, "straight"))

                        i += 3

                        continue

        merged.append(runs[i])

        i += 1

    return merged



def simplify_points_by_tags(
    points: List[Point],
    tags: List[Any],
    curve_epsilon_px: float = 1e-6,
) -> Tuple[List[Point], List[Any]]:
    """
    - "straight" runs are collapsed to just their first and last point
      (exactly 2 points), full stop -- no epsilon involved.
    - any other tag ("curve", "corner", ...) is left as-is, aside from a
      near-zero-epsilon DPHull pass that only removes truly redundant
      (near-duplicate) points, so arcs stay intact for later G2/G3 fitting.
    """
    n = len(points)
    if n < 3:
        return list(points), list(tags)

    runs = _split_into_runs(tags)

    runs = merge_straight_runs(points, runs)


    print(f"Runs: {runs}")

    
    out_pts: List[Point] = []
    out_tags: List[Any] = []

    for start, end, tag in runs:
        run_pts = points[start:end + 1]

        if len(run_pts) <= 2:
            seg_out = run_pts
        elif tag == "straight":
            seg_out = [run_pts[0], run_pts[-1]]
        else:
            seg_out = DPHull(run_pts, curve_epsilon_px).run()

        if out_pts and seg_out and out_pts[-1] == seg_out[0]:
            seg_out = seg_out[1:]

        out_pts.extend(seg_out)
        out_tags.extend([tag] * len(seg_out))


    for start, end, tag in runs:
        if tag == "straight":
            print(
                f"Straight run: {start}-{end}, "
                f"length={end-start+1}"
            )
            
    return out_pts, out_tags


def simplify_contour_by_tags(
    contour: Contour,
    curve_epsilon_mm: float,
    pixels_per_mm: float = 1.0,
    min_segment_mm: float = 0.02,
    validate: bool = True,
) -> Tuple[Contour, SimplifyReport]:
    """Same contract as simplify_contour(), but uses per-point tags
    (contour.metadata, e.g. from classify_points_straight_curve_corner) to
    drive simplification instead of a single global epsilon:
      - "straight" runs -> forced down to their 2 endpoints
      - everything else -> left alone (near-zero epsilon cleanup only)
    Falls back to plain simplify_contour() if tags are missing/mismatched.
    """
    notes: List[str] = []
    n_in = len(contour.points)

    if contour.metadata is None or len(contour.metadata) != n_in:
        notes.append("no per-point tags on this contour; falling back to plain DPHull")
        return simplify_contour(
            contour, epsilon_mm=curve_epsilon_mm, pixels_per_mm=pixels_per_mm,
            min_segment_mm=min_segment_mm, validate=validate,
        )

    curve_epsilon_px = curve_epsilon_mm * pixels_per_mm
    min_seg_px = min_segment_mm * pixels_per_mm

    pts, meta, dropped = dedupe_points(
        contour.points, contour.closed, min_seg_px, contour.metadata
    )
    if dropped:
        notes.append(f"dropped {dropped} near-duplicate point(s) (< {min_segment_mm} mm apart)")

    if len(pts) < (3 if contour.closed else 2):
        notes.append("too few points after cleanup; returning as-is")
        report = SimplifyReport(contour.contour_id, n_in, len(pts), dropped,
                                 False, False, True, notes)
        return Contour(pts, contour.closed, contour.is_hole, contour.contour_id, meta), report

    intersects_before = has_self_intersections(pts, contour.closed) if validate else False
    if intersects_before:
        notes.append(
            "input contour is self-intersecting (likely a Path-Offsetting "
            "artifact at a sharp corner) -- simplifying anyway, but flag "
            "this contour for review; consider fixing the offset join type"
        )

    if contour.closed:
        pts_r, tags_r = _rotate_to_run_boundary(pts, meta)
    else:
        pts_r, tags_r = pts, meta

    try:
        out_pts, out_tags = simplify_points_by_tags(pts_r, tags_r, curve_epsilon_px)
    except Exception as exc:
        notes.append(f"tag-aware simplification failed ({exc!r}); falling back to cleaned-up original")
        report = SimplifyReport(contour.contour_id, n_in, len(pts), dropped,
                                 intersects_before, intersects_before, True, notes)
        return Contour(pts, contour.closed, contour.is_hole, contour.contour_id, meta), report

    intersects_after = has_self_intersections(out_pts, contour.closed) if validate else False
    fell_back = False
    if intersects_after and not intersects_before:
        notes.append(
            "tag-aware simplification introduced a self-intersection that "
            "wasn't in the input -- discarding it and keeping the "
            "cleaned-up original for this contour"
        )
        out_pts, out_tags = pts, meta
        fell_back = True

    report = SimplifyReport(
        contour_id=contour.contour_id,
        input_points=n_in,
        output_points=len(out_pts),
        dropped_near_duplicates=dropped,
        had_self_intersections_before=intersects_before,
        had_self_intersections_after=intersects_after,
        fell_back_to_original=fell_back,
        notes=notes,
    )
    return Contour(out_pts, contour.closed, contour.is_hole, contour.contour_id, out_tags), report


def simplify_pipeline_by_tags(
    contours: Sequence[Contour],
    curve_epsilon_mm: float,
    pixels_per_mm: float = 1.0,
    min_segment_mm: float = 0.02,
    validate: bool = True,
) -> Tuple[List[Contour], List[SimplifyReport]]:
    out_contours: List[Contour] = []
    reports: List[SimplifyReport] = []
    for c in contours:
        simplified, report = simplify_contour_by_tags(
            c, curve_epsilon_mm, pixels_per_mm, min_segment_mm, validate
        )
        out_contours.append(simplified)
        reports.append(report)
    return out_contours, reports


# 5. Batch driver over every contour in an engraving
def simplify_pipeline(
    contours: Sequence[Contour],
    epsilon_mm: float,
    pixels_per_mm: float = 1.0,
    min_segment_mm: float = 0.02,
    validate: bool = True,
) -> Tuple[List[Contour], List[SimplifyReport]]:
    out_contours: List[Contour] = []
    reports: List[SimplifyReport] = []
    for c in contours:
        simplified, report = simplify_contour(
            c, epsilon_mm, pixels_per_mm, min_segment_mm, validate
        )
        out_contours.append(simplified)
        reports.append(report)
    return out_contours, reports