from math import hypot, acos, degrees, sqrt
import random
import matplotlib.pyplot as plt

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
 
 
def fit_arc_to_curved_segment(
    curved_segment,
    max_arc_error=0.15,
    min_arc_points=6,
):

    points = curved_segment["points"]

    if len(points) < min_arc_points:
        return None

    start = points[0]
    mid = points[len(points) // 2]
    end = points[-1]

    circle = circle_from_3_points(
        start,
        mid,
        end
    )

    if circle is None:
        return None

    max_error = max(
        point_circle_error(
            point,
            circle
        )
        for point in points
    )

    if max_error > max_arc_error:
        return None

    turn_signs = []

    for i in range(1, len(points) - 1):

        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]

        v1 = (
            p1[0] - p0[0],
            p1[1] - p0[1],
        )

        v2 = (
            p2[0] - p1[0],
            p2[1] - p1[1],
        )

        cross = (
            v1[0] * v2[1]
            - v1[1] * v2[0]
        )

        if abs(cross) < 1e-9:
            continue

        turn_signs.append(
            1 if cross > 0 else -1
        )

    if len(turn_signs) < 3:
        return None

    positive = sum(
        1 for s in turn_signs if s > 0
    )

    negative = sum(
        1 for s in turn_signs if s < 0
    )

    dominant = max(
        positive,
        negative
    )

    consistency = (
        dominant / len(turn_signs)
    )

    if consistency < 0.90:
        return None


    first = points[0]
    last = points[-1]

    chord = hypot(
        last[0] - first[0],
        last[1] - first[1],
    )

    radius = circle[2]

    if radius < 1e-9:
        return None

    if radius > max(
        1000.0,
        chord * 50.0
    ):
        return None

    return {
        "type": "arc",
        "command": arc_direction(
            start,
            mid,
            end
        ),
        "start": start,
        "end": end,
        "center": (
            circle[0],
            circle[1]
        ),
        "radius": circle[2],
        "points": points,
        "max_error": max_error,
        "turning_consistency": consistency,
    }

    
def build_path_representations(
    simplified_offset_paths,
    depth=-2.0,
    clearance_height=5.0,
    close_tol=1e-6,
    tags_per_path=None,
    verbose=True,
):

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
 
 
def build_fixed_path_options(paths_info):
    fixed_options = []

    for node in paths_info:
        points = list(node["points"])

        fixed_options.append({
            "path_id": node["id"],
            "type": node["type"],
            "depth": node["depth"],
            "clearance_height": node["clearance_height"],
            "sharp_corners": node.get("sharp_corners", []),
            "curved_segments": node.get("curved_segments", []),
            "straight_segments": node.get("straight_segments", []),
            "curve_tag_lookup": node.get("curve_tag_lookup"),
            "option_id": 0,
            "direction": (
                "forward"
                if node["type"] == "open"
                else "original"
            ),
            "entry_index": 0,
            "points": points,
            "start": points[0],
            "end": points[-1],
        })

    return fixed_options


def vector(p1, p2):
    return p2[0] - p1[0], p2[1] - p1[1]
 
 
def compute_direction_penalty(node_a, node_b):
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
 

DEFAULT_CONFIG = {
    "rapid_feed": 5000,
    "clearance_height": 5.0,
    "w_distance": 1.0,
    "w_direction": 1.5,
}
 
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

def genetic_algorithm(
    initial_route,
    cost_matrix,
    population_size=120,
    generations=250,
    mutation_rate=0.30,
    elite_ratio=0.05,
    tournament_size=4,
    n_workers=None,
    patience=40,
    verbose=True
):

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
    best_cost = route_cost(
        best_route,
        cost_matrix
    )

    history = []
    generations_without_improvement = 0

    for generation in range(generations):
        costs = {
            tuple(route): route_cost(
                route,
                cost_matrix
            )
            for route in population
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

def two_opt_swap(route, i, k):
    return (
        route[:i]
        + route[i:k + 1][::-1]
        + route[k + 1:]
    )
 
 
def _local_window_cost(route, cost_matrix, i, k):
    total = 0.0
 
    for t in range(i - 1, k + 1):
        total += get_cost(
            cost_matrix,
            route[t],
            route[t + 1]
        )
 
    return total
 
 
def two_opt_plus_plus(
    route,
    cost_matrix,
    candidate_limit=12,
    max_iterations=20
):
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

def refresh_selected_path_geometry(
    optimized_paths
):
    for path in optimized_paths:
        points = path["points"]
 
        path["straight_segments"] = (
            detect_straight_segments(points)
        )
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
 
    fixed_nodes = build_fixed_path_options(
        paths_info
    )

    cost_matrix = build_cost_matrix(
        fixed_nodes,
        config
    )
 
    initial_route = cheapest_insertion(cost_matrix)
 
    if verbose:
        print(
            "Initial fixed-matrix cost:",
            route_cost(initial_route, cost_matrix)
        )
 
    initial_cost = route_cost(
        initial_route,
        cost_matrix
    )

    best_route_ga, best_cost_ga, ga_history = (
        genetic_algorithm(
            initial_route=initial_route,
            cost_matrix=cost_matrix,
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
        print("Best GA fixed-matrix cost:", best_cost_ga)

    final_route_candidate, final_cost_fixed = (
        two_opt_plus_plus(
            best_route_ga,
            cost_matrix,
            candidate_limit=two_opt_defaults["candidate_limit"],
            max_iterations=two_opt_defaults["max_iterations"],
        )
    )

    if verbose:
        print(
            "Fixed-matrix cost after 2-Opt++:",
            final_cost_fixed
        )

    if final_cost_fixed <= best_cost_ga:
        final_route = final_route_candidate
        final_cost = final_cost_fixed
    else:
        if verbose:
            print(
                "2-Opt++ result was worse under the fixed cost "
                f"({final_cost_fixed:.6f} > {best_cost_ga:.6f}); "
                "keeping the GA route instead."
            )
        final_route = best_route_ga
        final_cost = best_cost_ga

    optimized_paths = [
        dict(fixed_nodes[path_id])
        for path_id in final_route
    ]

    if verbose:
        print("Final fixed-matrix cost:", final_cost)

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
        initial_options = [
            fixed_nodes[path_id]
            for path_id in initial_route
        ]
        ga_options = [
            fixed_nodes[path_id]
            for path_id in best_route_ga
        ]
        final_options = optimized_paths

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
        print("FIXED-COST PATH OPTIMIZATION EVALUATION")
        print("=" * 60)

        print("\n1. Objective cost")
        print(f"Initial route      : {initial_cost:.6f}")
        print(f"After GA           : {best_cost_ga:.6f}")
        print(f"After 2-Opt++      : {final_cost:.6f}")
        print(
            "Final reduction   : "
            f"{percentage_reduction(initial_cost, final_cost):.2f}%"
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
            plt.ylabel("Best Fixed-Matrix Cost")
            plt.title("GA Convergence with Fixed Transition Cost")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    elif make_plot:
        plt.figure(figsize=(9, 6))
        plt.plot(ga_history)
        plt.xlabel("Generation")
        plt.ylabel("Best Fixed-Matrix Cost")
        plt.title("GA Convergence with Fixed Transition Cost")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
 
    return final_route, optimized_paths

if __name__ == "__main__":
    from engine import pathOffset as offset
 
    final_route, optimized_paths = optimize_paths_advanced(
        offset.offset_paths,
        make_plot=True,
        verbose=True,
    )