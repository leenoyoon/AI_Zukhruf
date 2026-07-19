"""
dphull_integration.py
--------------------------------------------------------------------------
The one file that connects `pathOffset.py`'s output to `dphull.py` /
`contour_pipeline.py`, so `pathOptimizstion.py` can consume simplified
paths instead of the raw shapely offset output.

Why this file exists instead of editing contour_pipeline.py directly:
`process_image_to_offset_paths()` (pathOffset.py) returns each path as
`list(polygon.exterior.coords)` -- shapely ALWAYS repeats the first
point at the end to close the ring. `contour_pipeline.Contour` expects
the opposite convention (no duplicated closing vertex; `closed=True`
implies the wrap-around edge). This adapter is the single place that
translates between those two conventions, so neither pathOffset.py nor
contour_pipeline.py has to know about the other's format.

Also note: by the time `pathOffset.py` builds `offset_paths`, every
point has *already* been multiplied by `pixel_to_mm` -- coordinates are
already in millimetres, not pixels. So `pixels_per_mm=1.0` here is
correct, not a placeholder.
--------------------------------------------------------------------------
"""

from typing import List, Tuple

from contour_pipeline import Contour, simplify_pipeline

Point = Tuple[float, float]


def simplify_offset_paths(
    offset_paths: List[List[Point]],
    epsilon_mm: float = 0.15,
    pixels_per_mm: float = 1.0,
    min_segment_mm: float = 0.02,
    verbose: bool = True,
) -> List[List[Point]]:
    """
    Drop-in replacement for `offset.offset_paths` (pathOffset.py's
    output) that runs DPhull simplification on every ring first.

    Input / output shape is identical to `process_image_to_offset_paths`'s
    return value: a list of closed rings, each ring a list of (x, y)
    tuples in mm with the first point repeated at the end -- so this
    can be swapped in for `offset.offset_paths` with no other code
    changes required downstream (build_path_representations,
    is_closed_path, etc. all keep working as-is).

    NOTE on holes: pathOffset.py's current `process_image_to_offset_paths`
    doesn't preserve cv2's contour hierarchy (parent/child, i.e. which
    rings are interior holes) -- unlike gcode_generator.py's version,
    which does. Every ring here is therefore treated as closed=True,
    is_hole=False for simplification purposes; that only affects which
    diagnostics get printed, not correctness of the simplified geometry
    itself. If hole/parent info gets added to pathOffset.py later, this
    function should take it as an extra parameter and pass it through
    to `Contour(is_hole=...)`.
    """
    contours = []
    for idx, ring in enumerate(offset_paths):
        pts = list(ring)
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]  # strip shapely's duplicated closing vertex
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
            pts.append(pts[0])  # restore the closing vertex callers expect
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
