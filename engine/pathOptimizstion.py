from math import hypot, acos, degrees, sqrt
from engine import pathOffset as offset
import random
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Path representation and geometric analysis
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
# 2. Arc fitting
# ============================================================

def circle_from_3_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2
    cd = (temp - x3**2 - y3**2) / 2

    det = (
        (x1 - x2) * (y2 - y3)
        - (x2 - x3) * (y1 - y2)
    )

    if abs(det) < 1e-9:
        return None

    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det

    radius = sqrt((cx - x1)**2 + (cy - y1)**2)

    return cx, cy, radius


def point_circle_error(point, circle):
    cx, cy, radius = circle
    x, y = point

    current_radius = sqrt((x - cx)**2 + (y - cy)**2)

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

    return "G02" if cross < 0 else "G03"


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
    offset_paths,
    depth=-2.0,
    clearance_height=5.0,
    close_tol=1e-6
):
    represented_paths = []

    for path_id, path in enumerate(offset_paths):
        if len(path) < 2:
            continue

        path_type = (
            "closed"
            if is_closed_path(path, close_tol)
            else "open"
        )

        item = {
            "id": path_id,
            "type": path_type,
            "points": path,
            "start": path[0],
            "end": path[-1],
            "length": path_length(path),
            "depth": depth,
            "clearance_height": clearance_height,
            "direction_options": [],
            "straight_segments": detect_straight_segments(path),
            "curved_segments": detect_curved_segments(path),
            "sharp_corners": detect_sharp_corners(path)
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

    return represented_paths


paths_info = build_path_representations(
    offset.offset_paths,
    depth=-2.0,
    clearance_height=5.0
)

print(f"Number of paths: {len(paths_info)}")


# ============================================================
# 3. CNC-aware transition cost
# ============================================================

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


def compute_jerk_penalty(node_a, node_b):
    path_a = node_a["points"]

    if len(path_a) < 2:
        return 0.0

    previous_direction = vector(path_a[-2], path_a[-1])
    transition_direction = vector(path_a[-1], node_b["start"])

    angle = angle_between(
        previous_direction,
        transition_direction
    )

    if angle is None:
        return 0.0

    if angle > 120:
        return 2.0

    if angle > 90:
        return 1.0

    if angle > 45:
        return 0.5

    return 0.0


def compute_corner_penalty(node_b):
    sharp_count = len(node_b.get("sharp_corners", []))
    return sharp_count * 0.2


def compute_transition_cost(node_a, node_b, config):
    xy_distance = distance(
        node_a["end"],
        node_b["start"]
    )

    air_time = (
        xy_distance /
        config["rapid_feed"]
    )

    z_move = abs(
        config["clearance_height"] -
        node_b["depth"]
    ) * 2

    direction_penalty = compute_direction_penalty(
        node_a,
        node_b
    )

    jerk_penalty = compute_jerk_penalty(
        node_a,
        node_b
    )

    corner_penalty = compute_corner_penalty(node_b)

    total_cost = (
        config["w_distance"] * xy_distance
        + config["w_air_time"] * air_time
        + config["w_z_move"] * z_move
        + config["w_direction"] * direction_penalty
        + config["w_jerk"] * jerk_penalty
        + config["w_corner"] * corner_penalty
    )

    return {
        "xy_distance": xy_distance,
        "air_time": air_time,
        "z_move": z_move,
        "direction_penalty": direction_penalty,
        "jerk_penalty": jerk_penalty,
        "corner_penalty": corner_penalty,
        "total_cost": total_cost
    }


config = {
    "rapid_feed": 5000,
    "clearance_height": 5.0,
    "w_distance": 1.0,
    "w_air_time": 100.0,
    "w_z_move": 0.5,
    "w_direction": 5.0,
    "w_jerk": 10.0,
    "w_corner": 2.0
}


# ============================================================
# 4. Fixed cost matrix used only to create the initial ordering
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


cost_matrix = build_cost_matrix(
    paths_info,
    config
)


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
# 5. Cheapest Insertion for the initial order
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


initial_route = cheapest_insertion(cost_matrix)

print(
    "Initial fixed-matrix cost:",
    route_cost(initial_route, cost_matrix)
)


# ============================================================
# 6. Generate machining options for every path
#    NEW: order is evaluated together with direction and entry point
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
        "straight_segments": node.get("straight_segments", [])
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


all_path_options = build_all_path_options(
    paths_info,
    sample_step=10,
    max_entry_candidates=4
)


# ============================================================
# 7. Dynamic Programming evaluator
#    NEW: for each proposed order, choose the best options jointly
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


dp_evaluator = RouteDPEvaluator(
    all_path_options,
    config
)

initial_dp_cost = dp_evaluator.cost(
    initial_route
)

print(
    "Initial DP-aware cost:",
    initial_dp_cost
)


# ============================================================
# 8. Genetic Algorithm helpers
# ============================================================

random.seed(42)


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
    dp_evaluator,
    tournament_size=3
):
    candidates = random.sample(
        population,
        tournament_size
    )

    return min(
        candidates,
        key=dp_evaluator.cost
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
    dp_evaluator,
    elite_size
):
    sorted_population = sorted(
        population,
        key=dp_evaluator.cost
    )

    return [
        route[:]
        for route in sorted_population[
            :elite_size
        ]
    ]


# ============================================================
# 9. Genetic Algorithm using the DP-aware fitness
# ============================================================

def genetic_algorithm(
    initial_route,
    dp_evaluator,
    population_size=120,
    generations=250,
    mutation_rate=0.30,
    elite_ratio=0.05,
    tournament_size=4
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
    best_cost = dp_evaluator.cost(
        best_route
    )

    history = []

    for generation in range(generations):
        population = sorted(
            population,
            key=dp_evaluator.cost
        )

        current_best = population[0]
        current_cost = dp_evaluator.cost(
            current_best
        )

        if current_cost < best_cost:
            best_route = current_best[:]
            best_cost = current_cost

        history.append(best_cost)

        new_population = get_elites(
            population,
            dp_evaluator,
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
                population,
                dp_evaluator,
                tournament_size
            )

            parent2 = tournament_selection(
                population,
                dp_evaluator,
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

        if (
            generation == 0
            or (generation + 1) % 25 == 0
        ):
            print(
                f"Generation {generation + 1}/"
                f"{generations} - Best cost: "
                f"{best_cost:.6f}"
            )

    return best_route, best_cost, history


best_route_ga, best_cost_ga, ga_history = (
    genetic_algorithm(
        initial_route=initial_route,
        dp_evaluator=dp_evaluator,
        population_size=120,
        generations=250,
        mutation_rate=0.30,
        elite_ratio=0.05,
        tournament_size=4
    )
)

print("Best GA DP-aware cost:", best_cost_ga)


# ============================================================
# 10. 2-Opt++ evaluated using the same DP-aware objective
# ============================================================

def two_opt_swap(route, i, k):
    return (
        route[:i]
        + route[i:k + 1][::-1]
        + route[k + 1:]
    )


def two_opt_plus_plus_dp(
    route,
    dp_evaluator,
    candidate_limit=20,
    max_iterations=30
):
    best_route = route[:]
    best_cost = dp_evaluator.cost(
        best_route
    )

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
                candidate = two_opt_swap(
                    best_route,
                    i,
                    k
                )

                candidate_cost = (
                    dp_evaluator.cost(
                        candidate
                    )
                )

                if candidate_cost < best_cost:
                    best_route = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

            if improved:
                break

        iteration += 1

    return best_route, best_cost


final_route, final_cost = (
    two_opt_plus_plus_dp(
        best_route_ga,
        dp_evaluator,
        candidate_limit=20,
        max_iterations=30
    )
)

print("Final DP-aware cost:", final_cost)


# ============================================================
# 11. Recover the final direction and entry/exit selections
# ============================================================

final_cost, optimized_paths = (
    dp_evaluator.solve(final_route)
)


def refresh_selected_path_geometry(
    optimized_paths
):
    for path in optimized_paths:
        points = path["points"]

        path["straight_segments"] = (
            detect_straight_segments(points)
        )

        path["curved_segments"] = (
            detect_curved_segments(points)
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


optimized_paths = (
    refresh_selected_path_geometry(
        optimized_paths
    )
)

optimized_paths = attach_arc_data_to_paths(
    optimized_paths,
    max_arc_error=0.15
)


# ============================================================
# 12. Evaluation
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


initial_dp_cost, initial_options = (
    dp_evaluator.solve(initial_route)
)

ga_dp_cost, ga_options = (
    dp_evaluator.solve(best_route_ga)
)

final_dp_cost, final_options = (
    dp_evaluator.solve(final_route)
)

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

long_jump_threshold = 20.0

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


# ============================================================
# 13. GA convergence plot
# ============================================================

plt.figure(figsize=(9, 6))
plt.plot(ga_history)
plt.xlabel("Generation")
plt.ylabel("Best DP-aware Cost")
plt.title("GA Convergence with Entry/Exit Dynamic Programming")
plt.grid(True)
plt.tight_layout()
plt.show()