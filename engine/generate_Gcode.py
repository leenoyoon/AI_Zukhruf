 
import math
import logging
from datetime import datetime
 
# Configure logging system
logging.basicConfig(level=logging.INFO, format="[GCode-Engine] %(levelname)s: %(message)s")
 
 
def _is_valid_point(pt):
    """
    Validates whether a point is a valid (X, Y) coordinate pair
    containing real numbers (not None, NaN, or Infinity).
    """
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return False
    x, y = pt[0], pt[1]
    if x is None or y is None:
        return False
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
        return False
    return True
 
 
def generate_gcode(
    optimized_paths,
    safe_z=5.0,
    retract_z=1.0,
    cut_depth=-3.0,
    step_down=1.0,
    feed_rate=800,
    plunge_rate=300,
    spindle_speed=12000,
):
    """
    Generates a robust, highly optimized, and minimal G-code string from input paths.
    """
    # 1. Sanitize and validate numeric parameters
    try:
        safe_z = abs(float(safe_z))
        retract_z = abs(float(retract_z))
        cut_depth = -abs(float(cut_depth)) if cut_depth != 0 else -1.0
        step_down = abs(float(step_down)) if step_down != 0 else abs(cut_depth)
        feed_rate = max(10.0, float(feed_rate))
        plunge_rate = max(10.0, float(plunge_rate))
        spindle_speed = max(1000, int(spindle_speed))
    except (ValueError, TypeError) as e:
        logging.warning(f"Invalid input parameter detected ({e}). Reverting to default safe values.")
        safe_z, retract_z, cut_depth, step_down = 5.0, 1.0, -3.0, 1.0
        feed_rate, plunge_rate, spindle_speed = 800.0, 300.0, 12000
 
    # retract_z must never exceed safe_z (it's meant to be a smaller, faster inter-pass lift)
    retract_z = min(retract_z, safe_z)
 
    # 2. Compute depth pass sequence
    total_depth = abs(cut_depth)
    step = abs(step_down)
    num_passes = math.ceil(total_depth / step) if step > 0 else 1
    depth_pass_list = [-min(p * step, total_depth) for p in range(1, num_passes + 1)]
 
    # 3. Construct Minimal G-code Header
    gcode = [
        "(--------------------------------------------------)",
        "( Robust CNC Wood Carving G-Code Generator         )",
        f"( Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}           )",
        f"( Total Cut Depth: {cut_depth:.2f} mm | Passes: {len(depth_pass_list)}    )",
        f"( Feed Rate: {feed_rate} mm/min | Plunge: {plunge_rate} mm/min )",
        "(--------------------------------------------------)",
        "G21 ; Millimeters",
        "G90 ; Absolute positioning",
        "G17 ; XY Plane",
        "G94 ; Feed rate per minute (mm/min), not per revolution",
        f"M3 S{spindle_speed}",
        "G4 P2 ; Dwell to let spindle reach full speed",
        f"G0 Z{safe_z:.3f}"
    ]
 
    if not optimized_paths:
        logging.warning("The 'optimized_paths' list is empty!")
        gcode.extend(["M5", f"G0 Z{safe_z:.3f}", "G0 X0 Y0", "M30"])
        return "\n".join(gcode)
 
    active_feed_rate = None
 
    def get_feed_suffix(target_feed):
        nonlocal active_feed_rate
        if active_feed_rate != target_feed:
            active_feed_rate = target_feed
            return f" F{target_feed}"
        return ""
 
    # 4. Process paths with optimized minimal motion
    processed_count = 0
    for path_idx, path in enumerate(optimized_paths):
        try:
            raw_points = path.get("points", []) if isinstance(path, dict) else path
            arc_segments = path.get("arc_segments", []) if isinstance(path, dict) else []
 
            valid_points = [pt for pt in raw_points if _is_valid_point(pt)]
            if len(valid_points) < 2:
                logging.warning(
                    f"Path #{path_idx + 1} skipped: fewer than 2 valid points "
                    f"({len(valid_points)} found)."
                )
                continue
 
            # Build rapid lookup for arcs by start index.
            # Warn on coordinate collisions instead of silently overwriting,
            # since a silent overwrite can corrupt arc detection on self-intersecting paths.
            point_to_idx = {}
            for i, pt in enumerate(valid_points):
                key = (round(pt[0], 4), round(pt[1], 4))
                if key in point_to_idx:
                    logging.warning(
                        f"Path #{path_idx + 1}: duplicate/near-duplicate point at index {i} "
                        f"collides with index {point_to_idx[key]}; arc matching may be affected."
                    )
                else:
                    point_to_idx[key] = i
 
            arc_map = {}
            for arc in arc_segments:
                if not isinstance(arc, dict):
                    continue
                arc_pts = arc.get("points", [])
                if len(arc_pts) >= 2:
                    st_k = (round(arc_pts[0][0], 4), round(arc_pts[0][1], 4))
                    end_k = (round(arc_pts[-1][0], 4), round(arc_pts[-1][1], 4))
                    if st_k in point_to_idx and end_k in point_to_idx:
                        s_idx, e_idx = point_to_idx[st_k], point_to_idx[end_k]
                        if s_idx < e_idx:
                            arc_map[s_idx] = (e_idx, arc)
 
            processed_count += 1
            start = valid_points[0]
 
            gcode.append(f"\n; --- Path {processed_count} ---")
            gcode.append(f"G0 X{start[0]:.3f} Y{start[1]:.3f}")
 
            # Execute depth passes
            for pass_idx, current_z in enumerate(depth_pass_list):
                # FIX: before every pass after the first, the tool is sitting at the
                # END of the path (not the start). It must retract and reposition to
                # the start point before plunging again, otherwise it carves a
                # diagonal gouge straight across the workpiece on open paths.
                if pass_idx > 0:
                    gcode.append(f"G0 Z{retract_z:.3f} ; inter-pass retract")
                    gcode.append(f"G0 X{start[0]:.3f} Y{start[1]:.3f}")
 
                f_plunge = get_feed_suffix(plunge_rate)
                gcode.append(f"G1 Z{current_z:.3f}{f_plunge}")
 
                # Traverse path points
                idx = 0
                while idx < len(valid_points) - 1:
                    f_cut = get_feed_suffix(feed_rate)
 
                    # Check for arc segment
                    if idx in arc_map:
                        next_idx, arc = arc_map[idx]
                        end_pt, center, start_pt, cmd = (
                            arc.get("end"),
                            arc.get("center"),
                            arc.get("start"),
                            arc.get("command", "G1")
                        )
 
                        if (
                            _is_valid_point(end_pt)
                            and _is_valid_point(center)
                            and _is_valid_point(start_pt)
                            and cmd in ["G2", "G3"]
                        ):
                            i_off = center[0] - start_pt[0]
                            j_off = center[1] - start_pt[1]
                            gcode.append(
                                f"{cmd} X{end_pt[0]:.3f} Y{end_pt[1]:.3f} I{i_off:.3f} J{j_off:.3f}{f_cut}"
                            )
                            idx = next_idx
                            continue
 
                    # Linear Cutting Motion
                    next_pt = valid_points[idx + 1]
                    gcode.append(f"G1 X{next_pt[0]:.3f} Y{next_pt[1]:.3f}{f_cut}")
                    idx += 1
 
            # Retract to safe height after completing all depth passes for the shape
            gcode.append(f"G0 Z{safe_z:.3f}")
 
        except Exception as err:
            logging.error(f"Error processing Path #{path_idx + 1}: {err}")
            gcode.append(f"G0 Z{safe_z:.3f} ; Safety retract")
 
    if processed_count == 0:
        logging.warning(
            "No paths were successfully processed; output contains only header/footer G-code."
        )
 
    # 5. End program sequence
    gcode.extend([
        "\n; --- End of Program ---",
        "M5",
        f"G0 Z{safe_z:.3f}",
        "G0 X0.000 Y0.000",
        "M30"
    ])
 
    logging.info(f"G-code generated successfully. Processed paths: {processed_count}/{len(optimized_paths)}")
    return "\n".join(gcode)
 
 
def generate_gcode_from_user_input(
    optimized_paths,
    user_settings: dict = None
):
    """
    Wrapper function to accept dynamic settings from UI, validate, and trigger G-code generation.
    """
    defaults = {
        "safe_z": 5.0,
        "retract_z": 1.0,
        "cut_depth": -3.0,
        "step_down": 1.0,
        "feed_rate": 800.0,
        "plunge_rate": 300.0,
        "spindle_speed": 12000,
    }
 
    settings = defaults.copy()
    if user_settings and isinstance(user_settings, dict):
        settings.update(user_settings)
 
    # Sanity bounds check and clamping
    safe_z = max(1.0, abs(float(settings.get("safe_z", 5.0))))
    retract_z = max(0.5, abs(float(settings.get("retract_z", 1.0))))
    cut_depth = -abs(float(settings.get("cut_depth", -3.0)))
    step_down = min(abs(cut_depth), max(0.1, abs(float(settings.get("step_down", 1.0)))))
    feed_rate = max(50.0, min(5000.0, float(settings.get("feed_rate", 800.0))))
    plunge_rate = max(20.0, min(2000.0, float(settings.get("plunge_rate", 300.0))))
    spindle_speed = max(3000, min(24000, int(settings.get("spindle_speed", 12000))))
 
    return generate_gcode(
        optimized_paths=optimized_paths,
        safe_z=safe_z,
        retract_z=retract_z,
        cut_depth=cut_depth,
        step_down=step_down,
        feed_rate=feed_rate,
        plunge_rate=plunge_rate,
        spindle_speed=spindle_speed
    )