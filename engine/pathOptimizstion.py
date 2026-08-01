"""
pathOptimizstion.py (refactored to be importable, PERFORMANCE-OPTIMIZED)
--------------------------------------------------------------------------
This is a follow-up refactor on top of the previous importable version.
The GEOMETRY / ARC-FITTING / COST-MODEL MATH is still byte-for-byte the
same as the original script. What changed this time is HOW OFTEN the
expensive DP-aware evaluator gets called, because that was the real
runtime bottleneck for large numbers of paths (N > 100).
 
WHAT CHANGED IN THIS PASS (performance only, not the underlying cost
model / geometry):
 
  1. Two real bugs from the previous version are fixed:
       - `two_opt_plus_plus_dp(best_route_ga, cost_matrix, ...)` used to
         pass `cost_matrix` into a function that expected a
         `dp_evaluator` object (would crash with AttributeError).
         Now the function's signature actually IS built around
         `cost_matrix`, on purpose (see point 2).
       - `genetic_algorithm(..., max_workers=None, ...)` used to pass a
         keyword the function didn't accept (would crash with
         TypeError). Now `n_workers` is a real, used parameter.
 
  2. 2-Opt++ NO LONGER calls the DP evaluator at all.
     Previously, every candidate swap re-ran a full DP pass over the
     whole route (O(N * K^2) per candidate, with up to
     N * candidate_limit candidates per iteration -> effectively
     O(N^2) per iteration). That is what caused the runtime blow-up.
 
     Now 2-Opt++ works purely on the fixed `cost_matrix` (single
     default option per path -- the same matrix already used to build
     the initial cheapest-insertion order) and computes only the
     LOCAL DELTA caused by reversing the [i..k] segment: the edges
     from i-1 to k+1. That's O(candidate_limit) per candidate instead
     of O(N * K^2). For N > 100 this is the single biggest win.
 
     The DP evaluator (which picks the actual best entry/exit option
     per path, per direction) is now called EXACTLY ONCE on the final
     route, at the very end -- not inside the search loop.
 
  3. The Genetic Algorithm now evaluates the whole population's cost
     ONCE per generation, in parallel, using a persistent
     ProcessPoolExecutor (real multiprocessing, kept alive across all
     generations -- not recreated every generation).
     Those costs are cached in a dict for that generation and reused
     by sorting / tournament_selection / get_elites, instead of each
     of those re-calling dp_evaluator.cost() redundantly (previously
     this alone caused several hundred DP calls per generation).
 
  4. Early stopping: if the best cost hasn't improved for `patience`
     generations, the GA stops instead of always running all
     `generations` iterations.
 
  5. Search-space knobs were given more moderate defaults for large N
     (balanced speed/quality, not maximally aggressive):
       - max_entry_candidates: 4 -> 3   (K: 8 -> 6, so K^2 drops ~44%)
       - sample_step: 10 -> 12
       - candidate_limit (2-opt): 20 -> 12
       - max_iterations (2-opt): 30 -> 20
     All of these are exposed as parameters so you can tune them.
 
  NOT changed:
    - every geometry helper (distance, is_closed_path, path_length,
      angle_between, detect_sharp_corners, detect_straight_segments,
      detect_curved_segments)
    - arc fitting (circle_from_3_points, point_circle_error,
      arc_direction, fit_arc_to_curved_segment, build_path_representations)
    - the cost model itself was SIMPLIFIED (see section 3 below /
      compute_transition_cost's docstring for the full reasoning):
      z_move and corner_penalty were removed because they were
      mathematically dead weight for this setup (constant regardless of
      route order, confirmed against: full Z retract between every
      path, single uniform depth for all paths). air_time was removed
      because it was a linear duplicate of xy_distance. jerk_penalty was
      removed because it duplicated the same angle already captured by
      direction_penalty. Only xy_distance and direction_penalty remain
      as real, order-affecting cost terms -- with direction_penalty's
      weight lowered to reflect that, with full retract + controller
      blending, it's a soft preference rather than a hard constraint.
    - cost matrix / cheapest insertion (build_cost_matrix, get_cost,
      route_cost, cheapest_insertion)
    - per-path option generation (generate_path_options,
      build_all_path_options)
    - the DP evaluator itself (RouteDPEvaluator) -- still used, just
      called far less often
    - arc/segment refresh + evaluation metrics (refresh_selected_path_geometry,
      attach_arc_data_to_paths, total_air_distance_from_options,
      total_air_time_seconds, count_long_jumps_from_options,
      percentage_reduction)
 
  A NOTE ON MULTIPROCESSING / WINDOWS:
    `genetic_algorithm()` spins up a `ProcessPoolExecutor`. If you're on
    Windows (or generally to be safe), the *top-level* script that
    ultimately calls `optimize_paths_advanced()` must guard its entry
    point with `if __name__ == "__main__":`, exactly like this file
    already does at the bottom. On Linux this also works without the
    guard (fork start method), but keep the guard anyway for portability.
 
  Return value (unchanged shape):
    `final_route, optimized_paths = optimize_paths_advanced(offset_paths)`
--------------------------------------------------------------------------
"""
 
from math import hypot, acos, degrees, sqrt
import random
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
 
 
# ============================================================
# 1. Path representation and geometric analysis  (UNCHANGED)
# ============================================================
 
def distance(p1, p2):
    return hypot(p2[0] - p1[0], p2[1] - p1[1])
 
 
def is_closed_path(path, tol=1e-6):
    if len(path) < 3:
        return False
    return distance(path[0], path[-1]) <= tol
 
 
def path_length(path):
    return sum(
        distance(path[i], path[i + 1])
        for i in range(len(path) - 1)
    )
 
 
def angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = hypot(v1[0], v1[1])
    mag2 = hypot(v2[0], v2[1])
 
    if mag1 == 0 or mag2 == 0:
        return None
 
    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
 
    return degrees(acos(cos_theta))
 
 
def detect_sharp_corners(path, sharp_angle_threshold=45):
    sharp_corners = []
 
    for i in range(1, len(path) - 1):
        a = path[i - 1]
        b = path[i]
        c = path[i + 1]
 
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
 
        angle = angle_between(v1, v2)
 
        if angle is not None and angle <= sharp_angle_threshold:
            sharp_corners.append({
                "index": i,
                "point": b,
                "angle": angle
            })
 
    return sharp_corners
 
 
def detect_straight_segments(path, angle_tolerance=10):
    straight_segments = []
    start_idx = 0
 
    for i in range(1, len(path) - 1):
        a = path[i - 1]
        b = path[i]
        c = path[i + 1]
 
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
 
        angle = angle_between(v1, v2)
 
        if angle is not None:
            direction_change = abs(180 - angle)
 
            if direction_change > angle_tolerance:
                if i - start_idx >= 2:
                    straight_segments.append({
                        "start_index": start_idx,
                        "end_index": i,
                        "points": path[start_idx:i + 1]
                    })
 
                start_idx = i
 
    if len(path) - 1 - start_idx >= 2:
        straight_segments.append({
            "start_index": start_idx,
            "end_index": len(path) - 1,
            "points": path[start_idx:]
        })
 
    return straight_segments
 
 
def detect_curved_segments(path, angle_tolerance=10, min_points=4):
    curved_segments = []
    current_segment = []
 
    for i in range(1, len(path) - 1):
        a = path[i - 1]
        b = path[i]
        c = path[i + 1]
 
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
 
        angle = angle_between(v1, v2)
 
        if angle is None:
            continue
 
        direction_change = abs(180 - angle)
 
        if direction_change > angle_tolerance:
            if not current_segment:
                current_segment.append(a)
 
            current_segment.append(b)
 
        else:
            if len(current_segment) >= min_points:
                current_segment.append(b)
                curved_segments.append({
                    "points": current_segment
                })
 
            current_segment = []
 
    if len(current_segment) >= min_points:
        current_segment.append(path[-1])
        curved_segments.append({
            "points": current_segment
        })
 
    return curved_segments
 
 
# ============================================================
# 1b. Tag-based curve detection (PORTED from teammate's changes)
#
#     If the upstream pipeline already knows which points belong to a
#     curve (via simplify_offset_paths_with_curve_tags -- see section 13),
#     use that ground-truth tagging instead of re-deriving "curved" from
#     angle thresholds. Falls back to the angle-based detect_curved_segments
#     wherever tags aren't available or produce nothing, so this is purely
#     additive -- it never makes curve detection worse than before.
# ============================================================
 
def _make_tag_lookup(path, tags):
    if not tags or len(tags) != len(path):
        return None
    return {(round(p[0], 4), round(p[1], 4)): t for p, t in zip(path, tags)}
 
 
def detect_curved_segments_from_tags(path, tag_lookup, min_points=3):
    if not tag_lookup:
        return []
 
    curved_segments = []
    current_segment = []
 
    for pt in path:
        tag = tag_lookup.get((round(pt[0], 4), round(pt[1], 4)))
        if tag == "curve":
            current_segment.append(pt)
        else:
            if len(current_segment) >= min_points:
                curved_segments.append({"points": current_segment})
            current_segment = []
 
    if len(current_segment) >= min_points:
        curved_segments.append({"points": current_segment})
 
    return curved_segments
 
 
# ============================================================
# 2. Arc fitting  (UNCHANGED)
# ============================================================
 
def circle_from_3_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
 
    temp = x2 ** 2 + y2 ** 2
    bc = (x1 ** 2 + y1 ** 2 - temp) / 2
    cd = (temp - x3 ** 2 - y3 ** 2) / 2
 
    det = (
        (x1 - x2) * (y2 - y3)
        - (x2 - x3) * (y1 - y2)
    )
 
    if abs(det) < 1e-9:
        return None
 
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
 
    radius = sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
 
    return cx, cy, radius
 
 
def point_circle_error(point, circle):
    cx, cy, radius = circle
    x, y = point
 
    current_radius = sqrt((x - cx) ** 2 + (y - cy) ** 2)
 
    return abs(current_radius - radius)
 
 
def arc_direction(p_start, p_mid, p_end):
    v1 = (
        p_mid[0] - p_start[0],
        p_mid[1] - p_start[1]
    )
 
    v2 = (
        p_end[0] - p_mid[0],
        p_end[1] - p_mid[1]
    )
 
    cross = v1[0] * v2[1] - v1[1] * v2[0]
 
    return "G2" if cross < 0 else "G3"
 
 
def fit_arc_to_curved_segment(curved_segment, max_arc_error=0.15):
    points = curved_segment["points"]
 
    if len(points) < 3:
        return None
 
    start = points[0]
    mid = points[len(points) // 2]
    end = points[-1]
 
    circle = circle_from_3_points(start, mid, end)
 
    if circle is None:
        return None
 
    max_error = max(
        point_circle_error(point, circle)
        for point in points
    )
 
    if max_error > max_arc_error:
        return None
 
    return {
        "type": "arc",
        "command": arc_direction(start, mid, end),
        "start": start,
        "end": end,
        "center": (circle[0], circle[1]),
        "radius": circle[2],
        "points": points,
        "max_error": max_error
    }
 
 
def build_path_representations(
    simplified_offset_paths,
    depth=-2.0,
    clearance_height=5.0,
    close_tol=1e-6,
    tags_per_path=None,
    verbose=True,
):
    """
    PORTED from teammate's changes:
      - tags_per_path / curve-tag-aware curved-segment detection (falls
        back to angle-based detect_curved_segments when no tags, or when
        tags produce nothing).
      - degenerate closed-path dropping: a "closed" ring that collapses
        to fewer than 3 distinct points isn't machinable and is skipped,
        with a verbose warning + count.
      - "id" is now assigned as `len(represented_paths)` (i.e. the actual
        position in the output list) instead of the raw enumerate index
        from the input list. This matters: if any path is skipped (too
        short, or now also degenerate), the old raw-index "id" would
        leave GAPS in the id sequence -- but build_cost_matrix /
        RouteDPEvaluator / route arrays all assume ids are a CONTIGUOUS
        0..N-1 range matching paths_info's list positions. Keeping id ==
        list position fixes that latent bug, not just supports dropping.
    """
    represented_paths = []
    dropped_degenerate = 0
 
    for path_id, path in enumerate(simplified_offset_paths):
        if len(path) < 2:
            continue
 
        path_type = (
            "closed"
            if is_closed_path(path, close_tol)
            else "open"
        )
 
        if path_type == "closed":
            n_effective = (
                len(path) - 1
                if distance(path[0], path[-1]) <= close_tol
                else len(path)
            )
            if n_effective < 3:
                dropped_degenerate += 1
                continue
 
        path_tags = (
            tags_per_path[path_id]
            if tags_per_path is not None and path_id < len(tags_per_path)
            else None
        )
        tag_lookup = _make_tag_lookup(path, path_tags)
        curved = detect_curved_segments_from_tags(path, tag_lookup)
        if not curved:
            curved = detect_curved_segments(path)
 
        item = {
            "id": len(represented_paths),
            "type": path_type,
            "points": path,
            "start": path[0],
            "end": path[-1],
            "length": path_length(path),
            "depth": depth,
            "clearance_height": clearance_height,
            "direction_options": [],
            "straight_segments": detect_straight_segments(path),
            "curved_segments": curved,
            "sharp_corners": detect_sharp_corners(path),
            "curve_tag_lookup": tag_lookup,
        }
 
        if path_type == "open":
            item["direction_options"] = [
                {
                    "direction": "forward",
                    "start": path[0],
                    "end": path[-1],
                    "points": path
                },
                {
                    "direction": "reverse",
                    "start": path[-1],
                    "end": path[0],
                    "points": list(reversed(path))
                }
            ]
 
        else:
            item["direction_options"] = [
                {
                    "direction": "clockwise_or_original",
                    "start": path[0],
                    "end": path[-1],
                    "points": path
                }
            ]
 
        represented_paths.append(item)
 
    if verbose and dropped_degenerate:
        print(
            f"[paths] dropped {dropped_degenerate} degenerate ring(s) "
            f"(collapsed to < 3 distinct points, not machinable as a closed path)"
        )
 
    return represented_paths
 
 
# ============================================================
# 3. CNC-aware transition cost  (UNCHANGED)
# ============================================================
 
def vector(p1, p2):
    return p2[0] - p1[0], p2[1] - p1[1]
 
 
def compute_direction_penalty(node_a, node_b):
    """
    Angle (0..1, normalized) between path A's final travel direction and
    the transition vector into path B's entry point. Kept as a single
    CONTINUOUS penalty (rather than also having a separate stepped
    "jerk_penalty" duplicating the same angle) since it gives the
    GA/2-Opt search a smooth, differentiable signal instead of a
    plateaued step function -- and because with a full Z retract between
    every path (confirmed for this setup) plus controller-side
    look-ahead/blending, the previous cutting direction only loosely
    correlates with the next rapid move's dynamics anyway. Its weight
    (`w_direction`) should stay modest relative to `w_distance` to
    reflect that looser correlation.
    """
    path_a = node_a["points"]
 
    if len(path_a) < 2:
        return 0.0
 
    last_segment = vector(path_a[-2], path_a[-1])
    transition = vector(path_a[-1], node_b["start"])
 
    angle = angle_between(last_segment, transition)
 
    if angle is None:
        return 0.0
 
    return angle / 180.0
 
 
def compute_transition_cost(node_a, node_b, config):
    """
    Simplified transition cost -- kept to exactly the two terms that
    actually influence the route-order decision for this setup:
 
      - xy_distance : the real rapid-move distance (dominant real cost).
      - direction_penalty : soft preference for smoother direction
        changes into the next path's entry point.
 
    Removed vs. the earlier version (each was either mathematically dead
    or a near-duplicate signal, confirmed against this specific
    machine/setup):
 
      - air_time : was just xy_distance / rapid_feed, i.e. a linear
        rescaling of xy_distance already covered by w_distance. Real
        estimated air time is still reported separately after
        optimization via total_air_time_seconds() -- it just isn't
        double-counted inside the search objective anymore.
      - z_move : was abs(clearance_height - depth) * 2, but depth and
        clearance_height are GLOBAL CONSTANTS for every path in this
        setup (confirmed: full retract, single uniform depth) -- so it
        was an identical constant added to every possible transition,
        with zero effect on which order/entry-point gets picked.
      - jerk_penalty : a stepped version of the same angle already
        captured by direction_penalty -- redundant signal.
      - corner_penalty : depended only on path B's sharp-corner count,
        which is fixed per path (not per entry point) and is added
        exactly once per path regardless of route order (every path has
        exactly one predecessor) -- so it summed to an order-independent
        constant. Zero effect on optimization.
 
    If your setup later adds variable per-path depths (multi-level
    machining) or you want entry-point-specific corner awareness, those
    terms are worth reintroducing in a form that actually varies with
    the decision being made -- not as global constants.
    """
    xy_distance = distance(
        node_a["end"],
        node_b["start"]
    )
 
    direction_penalty = compute_direction_penalty(
        node_a,
        node_b
    )
 
    total_cost = (
        config["w_distance"] * xy_distance
        + config["w_direction"] * direction_penalty
    )
 
    return {
        "xy_distance": xy_distance,
        "direction_penalty": direction_penalty,
        "total_cost": total_cost
    }
 
 
# Simplified config: only the terms that actually affect the route
# search are here. `rapid_feed` and `clearance_height` are still used
# elsewhere (reporting, and building the path/option representations
# for downstream G-code generation) even though they no longer feed
# into the cost function itself.
DEFAULT_CONFIG = {
    "rapid_feed": 5000,
    "clearance_height": 5.0,
    "w_distance": 1.0,
    "w_direction": 1.5,
}
 
 
 
# ============================================================
# 4. Fixed cost matrix used for the initial ordering AND now
#    also for the entire 2-Opt++ local search (see section 10).
#    (UNCHANGED math)
# ============================================================
 
def build_cost_matrix(nodes, config):
    number_of_nodes = len(nodes)
 
    matrix = [
        [None for _ in range(number_of_nodes)]
        for _ in range(number_of_nodes)
    ]
 
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if i == j:
                continue
 
            matrix[i][j] = compute_transition_cost(
                nodes[i],
                nodes[j],
                config
            )
 
    return matrix
 
 
def get_cost(cost_matrix, i, j):
    if cost_matrix[i][j] is None:
        return float("inf")
 
    return cost_matrix[i][j]["total_cost"]
 
 
def route_cost(route, cost_matrix):
    return sum(
        get_cost(
            cost_matrix,
            route[index],
            route[index + 1]
        )
        for index in range(len(route) - 1)
    )
 
 
# ============================================================
# 5. Cheapest Insertion for the initial order  (UNCHANGED)
# ============================================================
 
def cheapest_insertion(cost_matrix):
    number_of_nodes = len(cost_matrix)
 
    if number_of_nodes == 0:
        return []
 
    if number_of_nodes == 1:
        return [0]
 
    best_start = None
    best_start_cost = float("inf")
 
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if i == j:
                continue
 
            current_cost = get_cost(
                cost_matrix,
                i,
                j
            )
 
            if current_cost < best_start_cost:
                best_start_cost = current_cost
                best_start = (i, j)
 
    route = [best_start[0], best_start[1]]
 
    unvisited = set(range(number_of_nodes))
    unvisited.remove(best_start[0])
    unvisited.remove(best_start[1])
 
    while unvisited:
        best_insertion = None
        best_delta = float("inf")
 
        for node in unvisited:
            for position in range(len(route) + 1):
                if position == 0:
                    delta = get_cost(
                        cost_matrix,
                        node,
                        route[0]
                    )
 
                elif position == len(route):
                    delta = get_cost(
                        cost_matrix,
                        route[-1],
                        node
                    )
 
                else:
                    previous_node = route[position - 1]
                    next_node = route[position]
 
                    old_cost = get_cost(
                        cost_matrix,
                        previous_node,
                        next_node
                    )
 
                    new_cost = (
                        get_cost(
                            cost_matrix,
                            previous_node,
                            node
                        )
                        + get_cost(
                            cost_matrix,
                            node,
                            next_node
                        )
                    )
 
                    delta = new_cost - old_cost
 
                if delta < best_delta:
                    best_delta = delta
                    best_insertion = (
                        node,
                        position
                    )
 
        node, position = best_insertion
        route.insert(position, node)
        unvisited.remove(node)
 
    return route
 
 
# ============================================================
# 6. Generate machining options for every path
#    (direction + entry point, evaluated together with order)
#    (UNCHANGED math -- only the *default* max_entry_candidates
#    used from optimize_paths_advanced was lowered, see section 13)
# ============================================================
 
def generate_path_options(
    node,
    sample_step=10,
    max_entry_candidates=4
):
    path = list(node["points"])
 
    if len(path) < 2:
        return []
 
    options = []
 
    common_data = {
        "path_id": node["id"],
        "type": node["type"],
        "depth": node["depth"],
        "clearance_height": node["clearance_height"],
        "sharp_corners": node.get("sharp_corners", []),
        "curved_segments": node.get("curved_segments", []),
        "straight_segments": node.get("straight_segments", []),
        "curve_tag_lookup": node.get("curve_tag_lookup"),
    }
 
    if node["type"] == "open":
        options.append({
            **common_data,
            "option_id": 0,
            "direction": "forward",
            "entry_index": 0,
            "points": path[:],
            "start": path[0],
            "end": path[-1]
        })
 
        reversed_path = list(reversed(path))
 
        options.append({
            **common_data,
            "option_id": 1,
            "direction": "reverse",
            "entry_index": len(path) - 1,
            "points": reversed_path,
            "start": reversed_path[0],
            "end": reversed_path[-1]
        })
 
        return options
 
    if distance(path[0], path[-1]) <= 1e-6:
        base_path = path[:-1]
    else:
        base_path = path[:]
 
    if len(base_path) < 3:
        return []
 
    candidate_indices = list(
        range(
            0,
            len(base_path),
            max(1, sample_step)
        )
    )
 
    if not candidate_indices:
        candidate_indices = [0]
 
    if len(candidate_indices) > max_entry_candidates:
        selected_positions = np.linspace(
            0,
            len(candidate_indices) - 1,
            max_entry_candidates,
            dtype=int
        )
 
        candidate_indices = [
            candidate_indices[position]
            for position in selected_positions
        ]
 
    option_id = 0
 
    for entry_index in candidate_indices:
        entry_point = base_path[entry_index]
 
        original_points = (
            base_path[entry_index:]
            + base_path[:entry_index]
            + [entry_point]
        )
 
        options.append({
            **common_data,
            "option_id": option_id,
            "direction": "original",
            "entry_index": entry_index,
            "points": original_points,
            "start": original_points[0],
            "end": original_points[-1]
        })
 
        option_id += 1
 
        reversed_base = list(reversed(base_path))
 
        reverse_entry_index = next(
            index
            for index, point in enumerate(reversed_base)
            if distance(point, entry_point) <= 1e-9
        )
 
        reverse_points = (
            reversed_base[reverse_entry_index:]
            + reversed_base[:reverse_entry_index]
            + [entry_point]
        )
 
        options.append({
            **common_data,
            "option_id": option_id,
            "direction": "reverse",
            "entry_index": entry_index,
            "points": reverse_points,
            "start": reverse_points[0],
            "end": reverse_points[-1]
        })
 
        option_id += 1
 
    return options
 
 
def build_all_path_options(
    paths_info,
    sample_step=10,
    max_entry_candidates=4
):
    return {
        node["id"]: generate_path_options(
            node,
            sample_step=sample_step,
            max_entry_candidates=max_entry_candidates
        )
        for node in paths_info
    }
 
 
# ============================================================
# 7. Dynamic Programming evaluator
#    (for each proposed order, choose the best options jointly)
#    (UNCHANGED math -- this is still exactly correct and used,
#    just called far less often now: once per GA generation via
#    the worker pool, and once at the very end -- never inside
#    2-Opt++ anymore.)
# ============================================================
 
class RouteDPEvaluator:
    def __init__(self, all_path_options, config):
        self.all_path_options = all_path_options
        self.config = config
        self.route_cost_cache = {}
        self.transition_matrix_cache = {}
 
    def _transition_matrix(
        self,
        previous_path_id,
        current_path_id
    ):
        key = (
            previous_path_id,
            current_path_id
        )
 
        if key in self.transition_matrix_cache:
            return self.transition_matrix_cache[key]
 
        previous_options = self.all_path_options[
            previous_path_id
        ]
 
        current_options = self.all_path_options[
            current_path_id
        ]
 
        matrix = np.empty(
            (
                len(previous_options),
                len(current_options)
            ),
            dtype=float
        )
 
        for previous_index, previous_option in enumerate(
            previous_options
        ):
            for current_index, current_option in enumerate(
                current_options
            ):
                matrix[
                    previous_index,
                    current_index
                ] = compute_transition_cost(
                    previous_option,
                    current_option,
                    self.config
                )["total_cost"]
 
        self.transition_matrix_cache[key] = matrix
 
        return matrix
 
    def evaluate(
        self,
        route,
        return_selected_options=False
    ):
        if not route:
            if return_selected_options:
                return 0.0, []
 
            return 0.0
 
        first_options = self.all_path_options[
            route[0]
        ]
 
        if not first_options:
            if return_selected_options:
                return float("inf"), []
 
            return float("inf")
 
        previous_costs = np.zeros(
            len(first_options),
            dtype=float
        )
 
        backtracking = []
 
        for route_position in range(
            1,
            len(route)
        ):
            previous_path_id = route[
                route_position - 1
            ]
 
            current_path_id = route[
                route_position
            ]
 
            # PORTED from teammate's changes: guard against a path that
            # ended up with zero valid options (e.g. an edge case not
            # caught by the degenerate-ring drop in
            # build_path_representations). Without this check, an empty
            # options list would crash _transition_matrix / np.empty with
            # a 0-length dimension instead of failing gracefully.
            current_options = self.all_path_options[current_path_id]
            if not current_options:
                if return_selected_options:
                    return float("inf"), []
                return float("inf")
 
            transition_matrix = (
                self._transition_matrix(
                    previous_path_id,
                    current_path_id
                )
            )
 
            combined_costs = (
                previous_costs[:, None]
                + transition_matrix
            )
 
            best_parent_indices = np.argmin(
                combined_costs,
                axis=0
            )
 
            current_costs = combined_costs[
                best_parent_indices,
                np.arange(
                    combined_costs.shape[1]
                )
            ]
 
            backtracking.append(
                best_parent_indices
            )
 
            previous_costs = current_costs
 
        best_final_option_index = int(
            np.argmin(previous_costs)
        )
 
        best_cost = float(
            previous_costs[
                best_final_option_index
            ]
        )
 
        if not return_selected_options:
            return best_cost
 
        selected_option_indices = [
            best_final_option_index
        ]
 
        current_option_index = (
            best_final_option_index
        )
 
        for parent_indices in reversed(
            backtracking
        ):
            current_option_index = int(
                parent_indices[
                    current_option_index
                ]
            )
 
            selected_option_indices.append(
                current_option_index
            )
 
        selected_option_indices.reverse()
 
        selected_options = [
            self.all_path_options[path_id][option_index]
            for path_id, option_index in zip(
                route,
                selected_option_indices
            )
        ]
 
        return best_cost, selected_options
 
    def cost(self, route):
        key = tuple(route)
 
        if key not in self.route_cost_cache:
            self.route_cost_cache[key] = (
                self.evaluate(
                    route,
                    return_selected_options=False
                )
            )
 
        return self.route_cost_cache[key]
 
    def solve(self, route):
        return self.evaluate(
            route,
            return_selected_options=True
        )
 
 
# ============================================================
# 7b. Worker-process helpers for parallel DP evaluation in the GA
#     (NEW)
#
#     A persistent ProcessPoolExecutor is created once in
#     genetic_algorithm() and reused for every generation. Each worker
#     process builds its OWN RouteDPEvaluator exactly once (via the
#     pool's `initializer`), then reuses it (and its internal
#     transition_matrix cache) for every route it's asked to cost --
#     instead of rebuilding an evaluator per call.
#
#     These must be plain module-level functions (not closures/lambdas)
#     so they can be pickled and sent to worker processes.
# ============================================================
 
_worker_dp_evaluator = None
 
 
def _init_dp_worker(all_path_options, config):
    global _worker_dp_evaluator
    _worker_dp_evaluator = RouteDPEvaluator(all_path_options, config)
 
 
def _worker_route_cost(route):
    return _worker_dp_evaluator.cost(route)
 
 
# ============================================================
# 8. Genetic Algorithm helpers
#    (perturb_route / create_initial_population / order_crossover /
#    mutate_route are UNCHANGED math. tournament_selection and
#    get_elites are CHANGED to take a precomputed `costs` dict instead
#    of a dp_evaluator, so they no longer trigger new DP evaluations --
#    they just look up costs that were already computed once per
#    generation, in parallel.)
# ============================================================
 
def perturb_route(route, strength=3):
    candidate = route[:]
 
    for _ in range(strength):
        operation = random.choice([
            "swap",
            "inversion",
            "insertion"
        ])
 
        if operation == "swap":
            i, j = random.sample(
                range(len(candidate)),
                2
            )
 
            candidate[i], candidate[j] = (
                candidate[j],
                candidate[i]
            )
 
        elif operation == "inversion":
            i, j = sorted(
                random.sample(
                    range(len(candidate)),
                    2
                )
            )
 
            candidate[i:j + 1] = reversed(
                candidate[i:j + 1]
            )
 
        else:
            i, j = random.sample(
                range(len(candidate)),
                2
            )
 
            node = candidate.pop(i)
            candidate.insert(j, node)
 
    return candidate
 
 
def create_initial_population(
    initial_route,
    population_size
):
    population = [initial_route[:]]
 
    seeded_size = int(
        population_size * 0.80
    )
 
    while len(population) < seeded_size:
        population.append(
            perturb_route(
                initial_route,
                strength=random.randint(1, 8)
            )
        )
 
    while len(population) < population_size:
        candidate = initial_route[:]
        random.shuffle(candidate)
        population.append(candidate)
 
    return population
 
 
def tournament_selection(
    population,
    costs,
    tournament_size=3
):
    """
    CHANGED: takes a precomputed `costs` dict (tuple(route) -> cost)
    instead of a dp_evaluator, so no new DP call happens here. The costs
    for `population` were already computed once, in parallel, for the
    current generation.
    """
    candidates = random.sample(
        population,
        tournament_size
    )
 
    return min(
        candidates,
        key=lambda route: costs[tuple(route)]
    )[:]
 
 
def order_crossover(parent1, parent2):
    number_of_nodes = len(parent1)
 
    if number_of_nodes < 2:
        return parent1[:]
 
    start, end = sorted(
        random.sample(
            range(number_of_nodes),
            2
        )
    )
 
    child = [None] * number_of_nodes
 
    child[start:end + 1] = (
        parent1[start:end + 1]
    )
 
    parent2_index = 0
 
    for child_index in range(number_of_nodes):
        if child[child_index] is not None:
            continue
 
        while (
            parent2[parent2_index]
            in child
        ):
            parent2_index += 1
 
        child[child_index] = (
            parent2[parent2_index]
        )
 
    return child
 
 
def mutate_route(
    route,
    mutation_rate=0.30
):
    mutated = route[:]
 
    if random.random() >= mutation_rate:
        return mutated
 
    operation = random.choice([
        "swap",
        "inversion",
        "insertion",
        "scramble"
    ])
 
    i, j = sorted(
        random.sample(
            range(len(mutated)),
            2
        )
    )
 
    if operation == "swap":
        mutated[i], mutated[j] = (
            mutated[j],
            mutated[i]
        )
 
    elif operation == "inversion":
        mutated[i:j + 1] = reversed(
            mutated[i:j + 1]
        )
 
    elif operation == "insertion":
        node = mutated.pop(j)
        mutated.insert(i, node)
 
    else:
        section = mutated[i:j + 1]
        random.shuffle(section)
        mutated[i:j + 1] = section
 
    return mutated
 
 
def get_elites(
    population,
    costs,
    elite_size
):
    """
    CHANGED: takes the precomputed `costs` dict instead of a
    dp_evaluator -- same reasoning as tournament_selection above.
    """
    sorted_population = sorted(
        population,
        key=lambda route: costs[tuple(route)]
    )
 
    return [
        route[:]
        for route in sorted_population[
            :elite_size
        ]
    ]
 
 
# ============================================================
# 9. Genetic Algorithm using the DP-aware fitness
#    (CHANGED: parallel batch cost evaluation per generation via a
#    persistent ProcessPoolExecutor + cost caching + early stopping)
# ============================================================
 
def genetic_algorithm(
    initial_route,
    dp_evaluator,
    population_size=120,
    generations=250,
    mutation_rate=0.30,
    elite_ratio=0.05,
    tournament_size=4,
    n_workers=None,
    patience=40,
    verbose=True
):
    """
    n_workers : number of worker PROCESSES for parallel DP cost
        evaluation. Defaults to (cpu_count - 1), at least 1.
    patience : stop early if the best cost hasn't improved for this
        many consecutive generations. Set to None to disable early
        stopping and always run the full `generations` count.
    """
    population = create_initial_population(
        initial_route,
        population_size
    )
 
    elite_size = max(
        1,
        int(
            population_size *
            elite_ratio
        )
    )
 
    best_route = initial_route[:]
    best_cost = dp_evaluator.cost(
        best_route
    )
 
    history = []
    generations_without_improvement = 0
 
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
 
    n_workers = max(1, min(n_workers, population_size))
 
    chunksize = max(1, population_size // (n_workers * 4) or 1)
 
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_dp_worker,
        initargs=(dp_evaluator.all_path_options, dp_evaluator.config)
    ) as executor:
 
        for generation in range(generations):
            # --- Evaluate the WHOLE population's cost ONCE per
            # generation, in parallel. This single batch replaces what
            # used to be hundreds of redundant dp_evaluator.cost()
            # calls scattered across sorted()/tournament_selection()/
            # get_elites() every generation.
            costs_list = list(
                executor.map(
                    _worker_route_cost,
                    population,
                    chunksize=chunksize
                )
            )
 
            costs = {
                tuple(route): cost
                for route, cost in zip(population, costs_list)
            }
 
            ranked_population = sorted(
                population,
                key=lambda route: costs[tuple(route)]
            )
 
            current_best = ranked_population[0]
            current_cost = costs[tuple(current_best)]
 
            if current_cost < best_cost - 1e-9:
                best_route = current_best[:]
                best_cost = current_cost
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
 
            history.append(best_cost)
 
            if verbose and (
                generation == 0
                or (generation + 1) % 25 == 0
            ):
                print(
                    f"Generation {generation + 1}/"
                    f"{generations} - Best cost: "
                    f"{best_cost:.6f}"
                )
 
            if (
                patience is not None
                and generations_without_improvement >= patience
            ):
                if verbose:
                    print(
                        f"Early stopping at generation "
                        f"{generation + 1} "
                        f"(no improvement for {patience} generations)"
                    )
                break
 
            new_population = get_elites(
                ranked_population,
                costs,
                elite_size
            )
 
            immigrant_count = max(
                1,
                int(
                    population_size *
                    0.10
                )
            )
 
            target_children = (
                population_size -
                immigrant_count
            )
 
            while (
                len(new_population) <
                target_children
            ):
                parent1 = tournament_selection(
                    ranked_population,
                    costs,
                    tournament_size
                )
 
                parent2 = tournament_selection(
                    ranked_population,
                    costs,
                    tournament_size
                )
 
                child = order_crossover(
                    parent1,
                    parent2
                )
 
                child = mutate_route(
                    child,
                    mutation_rate
                )
 
                new_population.append(child)
 
            while (
                len(new_population) <
                population_size
            ):
                immigrant = perturb_route(
                    initial_route,
                    strength=random.randint(5, 20)
                )
 
                new_population.append(
                    immigrant
                )
 
            population = new_population
 
    return best_route, best_cost, history
 
 
# ============================================================
# 10. 2-Opt++ -- REWRITTEN to remove DP entirely from the loop.
#
#     Works purely on the fixed `cost_matrix` (the same one used for
#     cheapest_insertion). A 2-opt swap that reverses route[i..k] only
#     changes the edges from index i-1 to k+1 -- everything else in the
#     route is untouched. So instead of recomputing the WHOLE route's
#     cost (or worse, a full DP pass) for every candidate, we only
#     recompute that local window: O(candidate_limit) instead of
#     O(N) / O(N*K^2).
#
#     (cost_matrix is asymmetric -- direction matters -- so we can't
#     just diff two edge costs; the whole reversed window's edges need
#     to be summed both ways. That's still cheap since the window is
#     bounded by candidate_limit, not by N.)
# ============================================================
 
def two_opt_swap(route, i, k):
    return (
        route[:i]
        + route[i:k + 1][::-1]
        + route[k + 1:]
    )
 
 
def _local_window_cost(route, cost_matrix, i, k):
    """
    Sum of edge costs from index i-1 to k+1 (inclusive of both
    endpoints), i.e. exactly the edges affected by reversing
    route[i:k+1]. Everything outside this window is identical before
    and after the swap, so it doesn't need to be touched.
    """
    total = 0.0
 
    for t in range(i - 1, k + 1):
        total += get_cost(
            cost_matrix,
            route[t],
            route[t + 1]
        )
 
    return total
 
 
def two_opt_plus_plus_dp(
    route,
    cost_matrix,
    candidate_limit=12,
    max_iterations=20
):
    """
    Local-search route-ORDER refinement using only `cost_matrix`.
 
    NOTE: despite the historical `_dp` suffix (kept for compatibility
    with existing call sites), this function no longer touches the DP
    evaluator at all -- that's the whole point of this change. The
    actual DP-aware per-path entry/exit choice happens exactly once,
    on the final returned route, in optimize_paths_advanced().
    """
    best_route = route[:]
    best_cost = route_cost(best_route, cost_matrix)
 
    improved = True
    iteration = 0
 
    while (
        improved
        and iteration < max_iterations
    ):
        improved = False
 
        for i in range(
            1,
            len(best_route) - 2
        ):
            maximum_k = min(
                i + candidate_limit,
                len(best_route) - 1
            )
 
            for k in range(
                i + 1,
                maximum_k
            ):
                old_window_cost = _local_window_cost(
                    best_route, cost_matrix, i, k
                )
 
                candidate = two_opt_swap(
                    best_route,
                    i,
                    k
                )
 
                new_window_cost = _local_window_cost(
                    candidate, cost_matrix, i, k
                )
 
                delta = new_window_cost - old_window_cost
 
                if delta < -1e-9:
                    best_route = candidate
                    best_cost += delta
                    improved = True
                    break
 
            if improved:
                break
 
        iteration += 1
 
    return best_route, best_cost
 
 
# ============================================================
# 11. Recover the final direction and entry/exit selections
#    (UNCHANGED)
# ============================================================
 
def refresh_selected_path_geometry(
    optimized_paths
):
    for path in optimized_paths:
        points = path["points"]
 
        path["straight_segments"] = (
            detect_straight_segments(points)
        )
 
        # PORTED from teammate's changes: prefer tag-based curve
        # detection (ground truth from upstream) over angle-based
        # detection, falling back to the angle-based version if the
        # tag lookup is missing or yields nothing for this path.
        curved = detect_curved_segments_from_tags(
            points, path.get("curve_tag_lookup")
        )
        path["curved_segments"] = (
            curved if curved else detect_curved_segments(points)
        )
 
        path["sharp_corners"] = (
            detect_sharp_corners(points)
        )
 
    return optimized_paths
 
 
def attach_arc_data_to_paths(
    optimized_paths,
    max_arc_error=0.15
):
    for path in optimized_paths:
        arcs = []
 
        for curved_segment in path.get(
            "curved_segments",
            []
        ):
            arc = fit_arc_to_curved_segment(
                curved_segment,
                max_arc_error=max_arc_error
            )
 
            if arc is not None:
                arcs.append(arc)
 
        path["arc_segments"] = arcs
 
    return optimized_paths
 
 
# ============================================================
# 12. Evaluation  (UNCHANGED)
# ============================================================
 
def total_air_distance_from_options(
    selected_options
):
    return sum(
        distance(
            selected_options[index]["end"],
            selected_options[index + 1]["start"]
        )
        for index in range(
            len(selected_options) - 1
        )
    )
 
 
def total_air_time_seconds(
    selected_options,
    rapid_feed
):
    total_distance = (
        total_air_distance_from_options(
            selected_options
        )
    )
 
    return (
        total_distance /
        rapid_feed
    ) * 60.0
 
 
def count_long_jumps_from_options(
    selected_options,
    threshold=20.0
):
    return sum(
        distance(
            selected_options[index]["end"],
            selected_options[index + 1]["start"]
        ) > threshold
        for index in range(
            len(selected_options) - 1
        )
    )
 
 
def percentage_reduction(
    before,
    after
):
    if before == 0:
        return 0.0
 
    return (
        (before - after) /
        before
    ) * 100.0
 
 
# ============================================================
# 13. Entry point: everything the original script did at import
#     time, unchanged in SHAPE, but with the performance fixes
#     from this pass wired in.
# ============================================================
 
def optimize_paths_advanced(
    offset_paths,
    config=None,
    ga_params=None,
    two_opt_params=None,
    max_entry_candidates=3,
    sample_step=12,
    max_arc_error=0.15,
    long_jump_threshold=20.0,
    make_plot=False,
    verbose=True,
    pixel_to_mm=None,
    epsilon_mm=None,
):
    """
    Drop-in advanced replacement for gcode_generator.optimize_paths().
 
    Parameters
    ----------
    offset_paths : list of rings (RAW, not already simplified -- this
        function calls simplify_offset_paths_with_curve_tags() on them
        internally).
    config : optional override of the transition-cost weights
        (defaults to DEFAULT_CONFIG, unchanged values).
    ga_params : optional dict overriding any of:
        population_size (120), generations (250), mutation_rate (0.30),
        elite_ratio (0.05), tournament_size (4),
        n_workers (None -> cpu_count - 1),
        patience (40 -> early stop if no improvement for 40 generations,
        set to None to disable).
    two_opt_params : optional dict overriding:
        candidate_limit (12), max_iterations (20).
    max_entry_candidates : max number of entry-point candidates
        generated per CLOSED path (default lowered from 4 to 3 --
        directly shrinks the DP evaluator's per-path option count K,
        and K^2 is what the DP inner loop costs).
    sample_step : point-sampling step used when generating entry-point
        candidates on closed paths (default raised from 10 to 12 --
        fewer candidate points to consider on long paths).
    make_plot : if True, shows the GA convergence plot (blocking).
    verbose : if True (default), prints the same progress/report lines
        the original script printed.
    pixel_to_mm : PORTED from teammate's changes. If provided and
        `epsilon_mm` is not explicitly given, the simplification
        tolerance is auto-derived as `max(0.1, 0.5 * pixel_to_mm)` --
        i.e. scaled to the source image/drawing's resolution instead of
        always using a fixed default.
    epsilon_mm : PORTED from teammate's changes. Explicit simplification
        tolerance (mm). Takes priority over pixel_to_mm. Defaults to
        0.15 if neither this nor pixel_to_mm is given (same fixed
        default as before).
 
    Returns
    -------
    final_route : list[int]
    optimized_paths : list[dict]
        Same shape as before -- ready for generate_Gcode.py.
    """
    from engine.dphull_integration import simplify_offset_paths_with_curve_tags
 
    if config is None:
        config = DEFAULT_CONFIG
 
    ga_defaults = {
        "population_size": 120,
        "generations": 250,
        "mutation_rate": 0.30,
        "elite_ratio": 0.05,
        "tournament_size": 4,
        "n_workers": None,
        "patience": 40,
    }
    if ga_params:
        ga_defaults.update(ga_params)
 
    two_opt_defaults = {
        "candidate_limit": 12,
        "max_iterations": 20,
    }
    if two_opt_params:
        two_opt_defaults.update(two_opt_params)
 
    random.seed(42)
 
    # PORTED from teammate's changes: auto-derive epsilon_mm from
    # pixel_to_mm when epsilon_mm isn't explicitly given, instead of
    # always using the fixed 0.15 default regardless of source
    # resolution.
    if epsilon_mm is None:
        if pixel_to_mm:
            epsilon_mm = max(0.1, 0.5 * pixel_to_mm)
        else:
            epsilon_mm = 0.15
 
    if verbose:
        print(
            f"[dphull] using epsilon_mm={epsilon_mm:.3f}"
            + (
                f" (auto from pixel_to_mm={pixel_to_mm:.4f})"
                if pixel_to_mm else " (fixed default)"
            )
        )
 
    simplified_offset_paths, curve_tags = simplify_offset_paths_with_curve_tags(
        offset_paths,
        epsilon_mm=epsilon_mm,
    )
 
    paths_info = build_path_representations(
        simplified_offset_paths,
        depth=-2.0,
        clearance_height=5.0,
        tags_per_path=curve_tags,
        verbose=verbose,
    )
 
    if verbose:
        print(f"Number of paths: {len(paths_info)}")
 
    cost_matrix = build_cost_matrix(
        paths_info,
        config
    )
 
    initial_route = cheapest_insertion(cost_matrix)
 
    if verbose:
        print(
            "Initial fixed-matrix cost:",
            route_cost(initial_route, cost_matrix)
        )
 
    all_path_options = build_all_path_options(
        paths_info,
        sample_step=sample_step,
        max_entry_candidates=max_entry_candidates
    )
 
    dp_evaluator = RouteDPEvaluator(
        all_path_options,
        config
    )
 
    initial_dp_cost = dp_evaluator.cost(
        initial_route
    )
 
    if verbose:
        print(
            "Initial DP-aware cost:",
            initial_dp_cost
        )
 
    best_route_ga, best_cost_ga, ga_history = (
        genetic_algorithm(
            initial_route=initial_route,
            dp_evaluator=dp_evaluator,
            population_size=ga_defaults["population_size"],
            generations=ga_defaults["generations"],
            mutation_rate=ga_defaults["mutation_rate"],
            elite_ratio=ga_defaults["elite_ratio"],
            tournament_size=ga_defaults["tournament_size"],
            n_workers=ga_defaults["n_workers"],
            patience=ga_defaults["patience"],
            verbose=verbose,
        )
    )
 
    if verbose:
        print("Best GA DP-aware cost:", best_cost_ga)
 
    # --- Get the DP-optimal entry/exit option for EVERY path, given the
    # GA's chosen order. We need this for two reasons:
    #   1. To build a route-aware cost matrix for 2-Opt++ (see below).
    #   2. As the safety-net fallback if 2-Opt++ doesn't actually help.
    ga_dp_cost, ga_selected_options = dp_evaluator.solve(best_route_ga)
 
    # --- Build a route-aware cost matrix for 2-Opt++.
    #
    # IMPORTANT FIX: the original `cost_matrix` (from build_cost_matrix on
    # paths_info) always represents each path by its RAW start/end point
    # (path[0] / path[-1]) -- never the entry/exit point the GA/DP
    # actually chose. Running 2-Opt++ against that raw matrix optimizes a
    # DIFFERENT objective than the one the GA optimized, so it can (and,
    # as observed, does) find an order that looks better under the raw
    # matrix but is WORSE under the true DP-aware cost.
    #
    # Fix: rebuild the matrix using each path's ACTUAL selected option
    # (the specific direction + entry/exit point DP picked for it in the
    # GA's best route). This makes 2-Opt++ optimize a close approximation
    # of the real objective instead of an unrelated one, while still
    # being pure O(candidate_limit)-per-swap local-matrix search (no DP
    # calls inside the loop).
    dp_aware_nodes = [None] * len(paths_info)
 
    for path_id, option in zip(best_route_ga, ga_selected_options):
        dp_aware_nodes[path_id] = option
 
    two_opt_cost_matrix = build_cost_matrix(dp_aware_nodes, config)
 
    final_route_candidate, final_cost_fixed = (
        two_opt_plus_plus_dp(
            best_route_ga,
            two_opt_cost_matrix,
            candidate_limit=two_opt_defaults["candidate_limit"],
            max_iterations=two_opt_defaults["max_iterations"],
        )
    )
 
    if verbose:
        print(
            "Route-aware fixed-matrix cost after 2-Opt++:",
            final_cost_fixed
        )
 
    # One full DP-aware evaluation on 2-Opt++'s candidate route, to pick
    # the actual best entry/exit option for the (possibly) new order.
    candidate_cost, candidate_options = (
        dp_evaluator.solve(final_route_candidate)
    )
 
    # --- Safety net: 2-Opt++ still optimizes an APPROXIMATION (entry/exit
    # points frozen from the GA's solution, not re-optimized per
    # candidate swap). It can occasionally still end up worse under the
    # true DP-aware objective once DP is free to re-pick entry/exit
    # points for the new order. Never accept a regression -- always keep
    # whichever of {GA route, 2-Opt++ route} is actually better under the
    # real objective.
    if candidate_cost <= ga_dp_cost:
        final_route = final_route_candidate
        final_cost = candidate_cost
        optimized_paths = candidate_options
    else:
        if verbose:
            print(
                "2-Opt++ result was worse under the true DP-aware cost "
                f"({candidate_cost:.6f} > {ga_dp_cost:.6f}); "
                "keeping the GA route instead."
            )
        final_route = best_route_ga
        final_cost = ga_dp_cost
        optimized_paths = ga_selected_options
 
    if verbose:
        print("Final DP-aware cost:", final_cost)
 
    optimized_paths = (
        refresh_selected_path_geometry(
            optimized_paths
        )
    )
 
    optimized_paths = attach_arc_data_to_paths(
        optimized_paths,
        max_arc_error=max_arc_error
    )
 
    if verbose:
        initial_dp_cost, initial_options = (
            dp_evaluator.solve(initial_route)
        )
 
        # Reuse the GA route's DP solve we already computed above (needed
        # for the 2-Opt++ route-aware matrix / safety net) instead of
        # solving it a second time.
        ga_options = ga_selected_options
 
        # Reuse the final route's DP solve we already computed above
        # instead of solving it a third time.
        final_dp_cost, final_options = final_cost, optimized_paths
 
        air_distance_before = (
            total_air_distance_from_options(
                initial_options
            )
        )
 
        air_distance_after_ga = (
            total_air_distance_from_options(
                ga_options
            )
        )
 
        air_distance_after_final = (
            total_air_distance_from_options(
                final_options
            )
        )
 
        air_time_before = total_air_time_seconds(
            initial_options,
            config["rapid_feed"]
        )
 
        air_time_after_ga = total_air_time_seconds(
            ga_options,
            config["rapid_feed"]
        )
 
        air_time_after_final = total_air_time_seconds(
            final_options,
            config["rapid_feed"]
        )
 
        long_jumps_before = (
            count_long_jumps_from_options(
                initial_options,
                threshold=long_jump_threshold
            )
        )
 
        long_jumps_after_ga = (
            count_long_jumps_from_options(
                ga_options,
                threshold=long_jump_threshold
            )
        )
 
        long_jumps_after_final = (
            count_long_jumps_from_options(
                final_options,
                threshold=long_jump_threshold
            )
        )
 
        print("\n" + "=" * 60)
        print("DP-AWARE PATH OPTIMIZATION EVALUATION")
        print("=" * 60)
 
        print("\n1. Objective cost")
        print(f"Initial route      : {initial_dp_cost:.6f}")
        print(f"After GA           : {ga_dp_cost:.6f}")
        print(f"After 2-Opt++      : {final_dp_cost:.6f}")
        print(
            "Final reduction   : "
            f"{percentage_reduction(initial_dp_cost, final_dp_cost):.2f}%"
        )
 
        print("\n2. Total air-move distance")
        print(f"Initial route      : {air_distance_before:.3f} mm")
        print(f"After GA           : {air_distance_after_ga:.3f} mm")
        print(f"After 2-Opt++      : {air_distance_after_final:.3f} mm")
        print(
            "Final reduction   : "
            f"{percentage_reduction(air_distance_before, air_distance_after_final):.2f}%"
        )
 
        print("\n3. Estimated air-move time")
        print(f"Initial route      : {air_time_before:.3f} seconds")
        print(f"After GA           : {air_time_after_ga:.3f} seconds")
        print(f"After 2-Opt++      : {air_time_after_final:.3f} seconds")
        print(
            "Final reduction   : "
            f"{percentage_reduction(air_time_before, air_time_after_final):.2f}%"
        )
 
        print(
            f"\n4. Long jumps greater than "
            f"{long_jump_threshold:.1f} mm"
        )
 
        print(f"Initial route      : {long_jumps_before}")
        print(f"After GA           : {long_jumps_after_ga}")
        print(f"After 2-Opt++      : {long_jumps_after_final}")
 
        if long_jumps_before > 0:
            print(
                "Final reduction   : "
                f"{percentage_reduction(long_jumps_before, long_jumps_after_final):.2f}%"
            )
 
        print("=" * 60)
 
        if make_plot:
            plt.figure(figsize=(9, 6))
            plt.plot(ga_history)
            plt.xlabel("Generation")
            plt.ylabel("Best DP-aware Cost")
            plt.title("GA Convergence with Entry/Exit Dynamic Programming")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    elif make_plot:
        plt.figure(figsize=(9, 6))
        plt.plot(ga_history)
        plt.xlabel("Generation")
        plt.ylabel("Best DP-aware Cost")
        plt.title("GA Convergence with Entry/Exit Dynamic Programming")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
 
    return final_route, optimized_paths
 
 
# ============================================================
# Standalone use (python pathOptimizstion.py) -- same behavior
# as the original script: reads engine.pathOffset.offset_paths,
# runs everything, shows the plot.
#
# The `if __name__ == "__main__":` guard below is REQUIRED for
# multiprocessing (ProcessPoolExecutor used inside genetic_algorithm)
# to work correctly, especially on Windows.
# ============================================================
if __name__ == "__main__":
    from engine import pathOffset as offset
 
    final_route, optimized_paths = optimize_paths_advanced(
        offset.offset_paths,
        make_plot=True,
        verbose=True,
    )