import os
import math


def optimize_paths(paths):
    if not paths:
        return []
    optimized = []
    current_pos = (0, 0)
    unvisited = [list(p) for p in paths]
    while unvisited:
        best_idx = 0
        best_point_idx = 0
        min_dist = float('inf')
        for i, path in enumerate(unvisited):
            for j, pt in enumerate(path):
                dist = math.dist(current_pos, pt)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
                    best_point_idx = j
        best_path = unvisited.pop(best_idx)
        if len(best_path) > 0 and best_path[0] == best_path[-1]:
            best_path_closed = best_path[:-1]
            best_path_rotated = best_path_closed[best_point_idx:] + best_path_closed[:best_point_idx]
            best_path_rotated.append(best_path_rotated[0])
            optimized.append(best_path_rotated)
            current_pos = best_path_rotated[-1]
        else:
            if best_point_idx == len(best_path) - 1:
                best_path.reverse()
            optimized.append(best_path)
            current_pos = best_path[-1]
    return optimized


def write_gcode(optimized_paths, output_path, depth=-3.0, step_down=1.0):
    gcode = [
        "G21 (Set units to mm)",
        "G90 (Absolute positioning)",
        "M3 S1000 (Spindle ON)",
    ]

    for path in optimized_paths:
        if not path:
            continue
        start_x, start_y = path[0]
        gcode.append(f"G0 X{start_x:.3f} Y{start_y:.3f} Z5.0")
        current_z = 0.0
        while current_z > depth:
            current_z -= step_down
            if current_z < depth:
                current_z = depth
            gcode.append(f"G1 Z{current_z:.3f} F200")
            for point in path[1:]:
                px, py = point
                gcode.append(f"G1 X{px:.3f} Y{py:.3f} F800")
        gcode.append("G0 Z5.0")

    gcode.append("M5 (Spindle OFF)")
    gcode.append("G0 X0 Y0 Z10 (Return to home)")
    gcode.append("M30 (End)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(gcode))

    print(f"Success! {len(optimized_paths)} paths saved to: {output_path}")