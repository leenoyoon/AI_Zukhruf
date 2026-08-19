import math
import time
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='[GCode-Engine] %(levelname)s: %(message)s')

def _is_valid_point(pt):
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return False
    x, y = (pt[0], pt[1])
    if x is None or y is None:
        return False
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
        return False
    return True

def _distance_2d(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def _arc_geometry(start_pt, center_pt, end_pt, clockwise):
    cx, cy = (center_pt[0], center_pt[1])
    sx, sy = (start_pt[0], start_pt[1])
    ex, ey = (end_pt[0], end_pt[1])
    radius = math.hypot(sx - cx, sy - cy)
    if radius < 1e-09:
        return (0.0, [(sx, sy), (ex, ey)])
    start_angle = math.atan2(sy - cy, sx - cx)
    end_angle = math.atan2(ey - cy, ex - cx)
    if clockwise:
        if end_angle >= start_angle:
            end_angle -= 2 * math.pi
    elif end_angle <= start_angle:
        end_angle += 2 * math.pi
    sweep = end_angle - start_angle
    arc_length = radius * abs(sweep)
    lo, hi = (end_angle, start_angle) if sweep < 0 else (start_angle, end_angle)
    extremes = [(sx, sy), (ex, ey)]
    k_min = math.ceil(lo / (math.pi / 2))
    k_max = math.floor(hi / (math.pi / 2))
    for k in range(k_min, k_max + 1):
        cardinal = k * (math.pi / 2)
        extremes.append((cx + radius * math.cos(cardinal), cy + radius * math.sin(cardinal)))
    return (arc_length, extremes)

def _update_bbox(stats, x, y, z=None):
    bmin, bmax = (stats['bbox_min'], stats['bbox_max'])
    bmin[0] = min(bmin[0], x)
    bmax[0] = max(bmax[0], x)
    bmin[1] = min(bmin[1], y)
    bmax[1] = max(bmax[1], y)
    if z is not None:
        bmin[2] = min(bmin[2], z)
        bmax[2] = max(bmax[2], z)

def _new_stats():
    return {'total_points_input': 0, 'total_points_valid': 0, 'linear_segments': 0, 'arc_segments': 0, 'plunge_moves': 0, 'cutting_length_mm': 0.0, 'plunge_length_mm': 0.0, 'rapid_xy_length_mm': 0.0, 'rapid_z_length_mm': 0.0, 'bbox_min': [math.inf, math.inf, math.inf], 'bbox_max': [-math.inf, -math.inf, -math.inf], 'warnings': []}

def _warn(stats, message):
    logging.warning(message)
    stats['warnings'].append(message)

def _record_error(stats, message):
    logging.error(message)
    stats['warnings'].append(f'ERROR: {message}')

def _estimate_machining_time(stats, feed_rate, plunge_rate, rapid_speed, dwell_seconds):
    rapid_total_mm = stats['rapid_xy_length_mm'] + stats['rapid_z_length_mm']
    cutting_time_min = stats['cutting_length_mm'] / feed_rate if feed_rate > 0 else 0.0
    plunge_time_min = stats['plunge_length_mm'] / plunge_rate if plunge_rate > 0 else 0.0
    rapid_time_min = rapid_total_mm / rapid_speed if rapid_speed > 0 else 0.0
    dwell_time_min = dwell_seconds / 60.0
    total_min = cutting_time_min + plunge_time_min + rapid_time_min + dwell_time_min
    total_seconds = round(total_min * 60)
    minutes, seconds = divmod(total_seconds, 60)
    return {'cutting_time_min': round(cutting_time_min, 3), 'plunge_time_min': round(plunge_time_min, 3), 'rapid_time_min_estimated': round(rapid_time_min, 3), 'dwell_time_min': round(dwell_time_min, 3), 'assumed_rapid_speed_mm_per_min': rapid_speed, 'total_time_min': round(total_min, 3), 'total_time_formatted': f'{minutes}m {seconds}s'}

def _estimate_material_removal(stats, num_passes, total_depth_mm, tool_diameter_mm, stepover_mm=None):
    if tool_diameter_mm is None or num_passes <= 0:
        return None
    effective_width_mm = stepover_mm if stepover_mm else tool_diameter_mm
    single_pass_length_mm = stats['cutting_length_mm'] / num_passes
    volume_mm3 = single_pass_length_mm * effective_width_mm * total_depth_mm
    return {'tool_diameter_mm': tool_diameter_mm, 'effective_width_used_mm': round(effective_width_mm, 3), 'basis': 'stepover (pocket/offset-fill)' if stepover_mm else 'tool diameter (single-contour assumption)', 'volume_mm3': round(volume_mm3, 1), 'volume_cm3': round(volume_mm3 / 1000.0, 3)}

def _estimate_mrr(feed_rate, step_down, tool_diameter_mm, warning_threshold_mm3_per_min=5000.0):
    if tool_diameter_mm is None:
        return None
    mrr = feed_rate * step_down * tool_diameter_mm
    return {'mrr_mm3_per_min': round(mrr, 1), 'warning_threshold_mm3_per_min': warning_threshold_mm3_per_min, 'high_mrr_warning': mrr > warning_threshold_mm3_per_min}

def _estimate_cost(total_time_min, machine_hourly_rate):
    if machine_hourly_rate is None:
        return None
    cost = total_time_min / 60.0 * machine_hourly_rate
    return {'machine_hourly_rate': machine_hourly_rate, 'estimated_cost': round(cost, 2)}

def _estimate_file_size(gcode_text, large_file_threshold_bytes=1000000):
    size_bytes = len(gcode_text.encode('utf-8'))
    return {'size_bytes': size_bytes, 'size_kb': round(size_bytes / 1024, 2), 'large_file_threshold_bytes': large_file_threshold_bytes, 'large_file_warning': size_bytes > large_file_threshold_bytes}

def _finalize_bbox(stats):
    bmin, bmax = (stats['bbox_min'], stats['bbox_max'])
    if math.isinf(bmin[0]):
        return None
    return {'x_min': round(bmin[0], 3), 'x_max': round(bmax[0], 3), 'y_min': round(bmin[1], 3), 'y_max': round(bmax[1], 3), 'z_min': round(bmin[2], 3), 'z_max': round(bmax[2], 3), 'width_mm': round(bmax[0] - bmin[0], 3), 'depth_mm': round(bmax[1] - bmin[1], 3), 'height_mm': round(bmax[2] - bmin[2], 3)}

def _build_report_comment_block(report):
    bbox = report['bounding_box']
    bbox_line = f"X[{bbox['x_min']:.2f}, {bbox['x_max']:.2f}] Y[{bbox['y_min']:.2f}, {bbox['y_max']:.2f}] Z[{bbox['z_min']:.2f}, {bbox['z_max']:.2f}]" if bbox else 'N/A (no geometry processed)'
    mt = report['machining_time_estimate']
    lines = [
        '(--------------------------------------------------)',
        '( Evaluation Summary / Analytics Report )',
        f"( Generation Time: {report['execution_time_ms']:.2f} s )",
        f"( Paths: {report['paths']['processed']}/{report['paths']['input']} processed | Points: {report['points']['valid']}/{report['points']['input']} valid )",
        f"( Segments: {report['segments']['linear']} linear (G1) | {report['segments']['arc']} arc (G2/G3) )",
        f"( Cutting Length: {report['lengths_mm']['cutting']:.2f} mm | Rapid XY (ordering): {report['lengths_mm']['rapid_xy']:.2f} mm | Rapid Z (safety lifts): {report['lengths_mm']['rapid_z']:.2f} mm )",
        f"( Est. Machining Time: {mt['total_time_formatted']} (assumes rapid @ {mt['assumed_rapid_speed_mm_per_min']:.0f} mm/min) )",
        f"( Bounding Box (stock size needed): {bbox_line} )",
        f"( Output File Size: {report['file_size']['size_kb']:.1f} KB )"
    ]
    if report.get('material_removal'):
        mr = report['material_removal']
        lines.append(f"( Est. Material Removed: {mr['volume_cm3']:.2f} cm^3 [basis: {mr['basis']}] )")
    if report.get('mrr'):
        mrr = report['mrr']
        flag = ' -- WARNING: high MRR, consider reducing feed rate' if mrr['high_mrr_warning'] else ''
        lines.append(f"( Peak MRR: {mrr['mrr_mm3_per_min']:.0f} mm^3/min{flag} )")
    if report.get('cost'):
        c = report['cost']
        lines.append(f"( Est. Machining Cost: {c['estimated_cost']:.2f} (@ {c['machine_hourly_rate']:.2f}/hr) )")
    lines.append('(--------------------------------------------------)')
    return lines

def _generate_gcode_core(optimized_paths, safe_z, retract_z, cut_depth, step_down, feed_rate, plunge_rate, spindle_speed, rapid_speed, tool_diameter_mm, stepover_mm, machine_hourly_rate, pipeline_start_perf=None):
    start_time = time.perf_counter()
    dwell_seconds = 2.0
    retract_z = min(retract_z, safe_z)
    total_depth = abs(cut_depth)
    step = abs(step_down)
    num_passes = math.ceil(total_depth / step) if step > 0 else 1
    depth_pass_list = [-min(p * step, total_depth) for p in range(1, num_passes + 1)]
    static_header = [
    '(--------------------------------------------------)',
    '( Robust CNC Wood Carving G-Code Generator )',
    f"( Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} )",
    f"( Total Cut Depth: {cut_depth:.2f} mm | Passes: {len(depth_pass_list)} )",
    f"( Feed Rate: {feed_rate} mm/min | Plunge: {plunge_rate} mm/min )",
    '(--------------------------------------------------)'
    ]
    init_lines = ['G21 ; Millimeters', 'G90 ; Absolute positioning', 'G17 ; XY Plane', 'G94 ; Feed rate per minute (mm/min), not per revolution', f'M3 S{spindle_speed}', 'G4 P2 ; Dwell to let spindle reach full speed', f'G0 Z{safe_z:.3f}']
    stats = _new_stats()
    cursor = [0.0, 0.0, 0.0]
    stats['rapid_z_length_mm'] += abs(safe_z - cursor[2])
    cursor[2] = safe_z

    def _finish(body_lines, processed_count, total_paths):
        stats['rapid_xy_length_mm'] += _distance_2d((cursor[0], cursor[1]), (0.0, 0.0))
        footer_lines = ['\n; --- End of Program ---', 'M5', f'G0 Z{safe_z:.3f}', 'G0 X0.000 Y0.000', 'M30']
        exec_seconds = time.perf_counter() - (pipeline_start_perf or start_time)
        mt = _estimate_machining_time(stats, feed_rate, plunge_rate, rapid_speed, dwell_seconds)
        report = {'execution_time_ms': round(exec_seconds, 3), 'paths': {'input': total_paths, 'processed': processed_count, 'skipped': total_paths - processed_count}, 'points': {'input': stats['total_points_input'], 'valid': stats['total_points_valid']}, 'segments': {'linear': stats['linear_segments'], 'arc': stats['arc_segments'], 'plunge_moves': stats['plunge_moves']}, 'lengths_mm': {'cutting': round(stats['cutting_length_mm'], 3), 'plunge': round(stats['plunge_length_mm'], 3), 'rapid_xy': round(stats['rapid_xy_length_mm'], 3), 'rapid_z': round(stats['rapid_z_length_mm'], 3)}, 'machining_time_estimate': mt, 'bounding_box': _finalize_bbox(stats), 'material_removal': _estimate_material_removal(stats, num_passes, total_depth, tool_diameter_mm, stepover_mm), 'mrr': _estimate_mrr(feed_rate, step, tool_diameter_mm), 'cost': _estimate_cost(mt['total_time_min'], machine_hourly_rate), 'warnings': list(stats['warnings']), 'file_size': {'size_bytes': 0, 'size_kb': 0.0, 'large_file_warning': False}}
        gcode_lines = static_header + _build_report_comment_block(report) + init_lines + body_lines + footer_lines
        temp_text = '\n'.join(gcode_lines)
        report['file_size'] = _estimate_file_size(temp_text)
        gcode_lines = static_header + _build_report_comment_block(report) + init_lines + body_lines + footer_lines
        final_text = '\n'.join(gcode_lines)
        report['total_gcode_lines'] = len(final_text.splitlines())
        logging.info(f"G-code generated successfully in {report['execution_time_ms']:.2f} s. Processed paths: {processed_count}/{total_paths}")
        return (final_text, report)
    if not optimized_paths:
        _warn(stats, "The 'optimized_paths' list is empty!")
        body_lines = []
        return _finish(body_lines, processed_count=0, total_paths=0)
    active_feed_rate = None

    def get_feed_suffix(target_feed):
        nonlocal active_feed_rate
        if active_feed_rate != target_feed:
            active_feed_rate = target_feed
            return f' F{target_feed}'
        return ''
    processed_count = 0
    body_lines = []
    for path_idx, path in enumerate(optimized_paths):
        try:
            raw_points = path.get('points', []) if isinstance(path, dict) else path
            arc_segments = path.get('arc_segments', []) if isinstance(path, dict) else []
            stats['total_points_input'] += len(raw_points)
            valid_points = [pt for pt in raw_points if _is_valid_point(pt)]
            if len(valid_points) < 2:
                _warn(stats, f'Path #{path_idx + 1} skipped: fewer than 2 valid points ({len(valid_points)} found).')
                continue
            stats['total_points_valid'] += len(valid_points)
            point_to_idx = {}
            for i, pt in enumerate(valid_points):
                key = (round(pt[0], 4), round(pt[1], 4))
                if key in point_to_idx:
                    pass
                else:
                    point_to_idx[key] = i
            arc_map = {}
            for arc in arc_segments:
                if not isinstance(arc, dict):
                    continue
                arc_pts = arc.get('points', [])
                if len(arc_pts) >= 2:
                    st_k = (round(arc_pts[0][0], 4), round(arc_pts[0][1], 4))
                    end_k = (round(arc_pts[-1][0], 4), round(arc_pts[-1][1], 4))
                    if st_k in point_to_idx and end_k in point_to_idx:
                        s_idx, e_idx = (point_to_idx[st_k], point_to_idx[end_k])
                        if s_idx < e_idx:
                            arc_map[s_idx] = (e_idx, arc)
            processed_count += 1
            start = valid_points[0]
            body_lines.append(f'\n; --- Path {processed_count} ---')
            body_lines.append(f'G0 X{start[0]:.3f} Y{start[1]:.3f}')
            stats['rapid_xy_length_mm'] += _distance_2d((cursor[0], cursor[1]), start)
            cursor[0], cursor[1] = (start[0], start[1])
            _update_bbox(stats, cursor[0], cursor[1], cursor[2])
            for pass_idx, current_z in enumerate(depth_pass_list):
                if pass_idx > 0:
                    body_lines.append(f'G0 Z{retract_z:.3f} ; inter-pass retract')
                    stats['rapid_z_length_mm'] += abs(retract_z - cursor[2])
                    cursor[2] = retract_z
                    body_lines.append(f'G0 X{start[0]:.3f} Y{start[1]:.3f}')
                    stats['rapid_xy_length_mm'] += _distance_2d((cursor[0], cursor[1]), start)
                    cursor[0], cursor[1] = (start[0], start[1])
                f_plunge = get_feed_suffix(plunge_rate)
                body_lines.append(f'G1 Z{current_z:.3f}{f_plunge}')
                stats['plunge_length_mm'] += abs(current_z - cursor[2])
                stats['plunge_moves'] += 1
                cursor[2] = current_z
                _update_bbox(stats, cursor[0], cursor[1], cursor[2])
                idx = 0
                while idx < len(valid_points) - 1:
                    f_cut = get_feed_suffix(feed_rate)
                    if idx in arc_map:
                        next_idx, arc = arc_map[idx]
                        end_pt, center, start_pt, cmd = (arc.get('end'), arc.get('center'), arc.get('start'), arc.get('command', 'G1'))
                        if _is_valid_point(end_pt) and _is_valid_point(center) and _is_valid_point(start_pt) and (cmd in ['G2', 'G3']):
                            i_off = center[0] - start_pt[0]
                            j_off = center[1] - start_pt[1]
                            body_lines.append(f'{cmd} X{end_pt[0]:.3f} Y{end_pt[1]:.3f} I{i_off:.3f} J{j_off:.3f}{f_cut}')
                            arc_len, extremes = _arc_geometry(start_pt, center, end_pt, cmd == 'G2')
                            stats['cutting_length_mm'] += arc_len
                            stats['arc_segments'] += 1
                            for ex, ey in extremes:
                                _update_bbox(stats, ex, ey, cursor[2])
                            cursor[0], cursor[1] = (end_pt[0], end_pt[1])
                            idx = next_idx
                            continue
                    next_pt = valid_points[idx + 1]
                    body_lines.append(f'G1 X{next_pt[0]:.3f} Y{next_pt[1]:.3f}{f_cut}')
                    stats['cutting_length_mm'] += _distance_2d((cursor[0], cursor[1]), next_pt)
                    stats['linear_segments'] += 1
                    cursor[0], cursor[1] = (next_pt[0], next_pt[1])
                    _update_bbox(stats, cursor[0], cursor[1], cursor[2])
                    idx += 1
            body_lines.append(f'G0 Z{safe_z:.3f}')
            stats['rapid_z_length_mm'] += abs(safe_z - cursor[2])
            cursor[2] = safe_z
        except Exception as err:
            _record_error(stats, f'Error processing Path #{path_idx + 1}: {err}')
            body_lines.append(f'G0 Z{safe_z:.3f} ; Safety retract')
            cursor[2] = safe_z
    if processed_count == 0:
        _warn(stats, 'No paths were successfully processed; output contains only header/footer G-code.')
    return _finish(body_lines, processed_count, total_paths=len(optimized_paths))

def _sanitize_core_params(safe_z, retract_z, cut_depth, step_down, feed_rate, plunge_rate, spindle_speed, rapid_speed):
    try:
        safe_z = abs(float(safe_z))
        retract_z = abs(float(retract_z))
        cut_depth = -abs(float(cut_depth)) if cut_depth != 0 else -1.0
        step_down = abs(float(step_down)) if step_down != 0 else abs(cut_depth)
        feed_rate = max(10.0, float(feed_rate))
        plunge_rate = max(10.0, float(plunge_rate))
        spindle_speed = max(1000, int(spindle_speed))
        rapid_speed = max(10.0, float(rapid_speed))
    except (ValueError, TypeError) as e:
        logging.warning(f'Invalid input parameter detected ({e}). Reverting to default safe values.')
        safe_z, retract_z, cut_depth, step_down = (5.0, 1.0, -3.0, 1.0)
        feed_rate, plunge_rate, spindle_speed, rapid_speed = (800.0, 300.0, 12000, 3000.0)
    return (safe_z, retract_z, cut_depth, step_down, feed_rate, plunge_rate, spindle_speed, rapid_speed)

def generate_gcode(optimized_paths, safe_z=5.0, retract_z=1.0, cut_depth=-3.0, step_down=1.0, feed_rate=800, plunge_rate=300, spindle_speed=12000, rapid_speed=3000.0, tool_diameter_mm=None, stepover_mm=None, machine_hourly_rate=None):
    params = _sanitize_core_params(safe_z, retract_z, cut_depth, step_down, feed_rate, plunge_rate, spindle_speed, rapid_speed)
    text, _report = _generate_gcode_core(optimized_paths, *params, tool_diameter_mm, stepover_mm, machine_hourly_rate)
    return text

def generate_gcode_with_report(optimized_paths, safe_z=5.0, retract_z=1.0, cut_depth=-3.0, step_down=1.0, feed_rate=800, plunge_rate=300, spindle_speed=12000, rapid_speed=3000.0, tool_diameter_mm=None, stepover_mm=None, machine_hourly_rate=None, pipeline_start_perf=None):
    params = _sanitize_core_params(safe_z, retract_z, cut_depth, step_down, feed_rate, plunge_rate, spindle_speed, rapid_speed)
    return _generate_gcode_core(optimized_paths, *params, tool_diameter_mm, stepover_mm, machine_hourly_rate, pipeline_start_perf=pipeline_start_perf)

def _resolve_user_settings(user_settings):
    defaults = {'safe_z': 5.0, 'retract_z': 1.0, 'cut_depth': -3.0, 'step_down': 1.0, 'feed_rate': 800.0, 'plunge_rate': 300.0, 'spindle_speed': 12000, 'rapid_speed': 3000.0, 'tool_diameter_mm': None, 'stepover_mm': None, 'machine_hourly_rate': None}
    settings = defaults.copy()
    if user_settings and isinstance(user_settings, dict):
        settings.update(user_settings)
    safe_z = max(1.0, abs(float(settings.get('safe_z', 5.0))))
    retract_z = max(0.5, abs(float(settings.get('retract_z', 1.0))))
    cut_depth = -abs(float(settings.get('cut_depth', -3.0)))
    step_down = min(abs(cut_depth), max(0.1, abs(float(settings.get('step_down', 1.0)))))
    feed_rate = max(50.0, min(5000.0, float(settings.get('feed_rate', 800.0))))
    plunge_rate = max(20.0, min(2000.0, float(settings.get('plunge_rate', 300.0))))
    spindle_speed = max(3000, min(24000, int(settings.get('spindle_speed', 12000))))
    rapid_speed = max(100.0, min(10000.0, float(settings.get('rapid_speed', 3000.0))))
    tool_diameter_mm = settings.get('tool_diameter_mm')
    if tool_diameter_mm is not None:
        tool_diameter_mm = max(0.1, min(50.0, float(tool_diameter_mm)))
    stepover_mm = settings.get('stepover_mm')
    if stepover_mm is not None:
        stepover_mm = max(0.01, float(stepover_mm))
    machine_hourly_rate = settings.get('machine_hourly_rate')
    if machine_hourly_rate is not None:
        machine_hourly_rate = max(0.0, float(machine_hourly_rate))
    return dict(safe_z=safe_z, retract_z=retract_z, cut_depth=cut_depth, step_down=step_down, feed_rate=feed_rate, plunge_rate=plunge_rate, spindle_speed=spindle_speed, rapid_speed=rapid_speed, tool_diameter_mm=tool_diameter_mm, stepover_mm=stepover_mm, machine_hourly_rate=machine_hourly_rate)

def generate_gcode_from_user_input(optimized_paths, user_settings: dict=None):
    return generate_gcode(optimized_paths, **_resolve_user_settings(user_settings))

def generate_gcode_from_user_input_with_report(optimized_paths, user_settings: dict=None, pipeline_start_perf=None):
    return generate_gcode_with_report(optimized_paths, pipeline_start_perf=pipeline_start_perf, **_resolve_user_settings(user_settings))

def print_gcode_report(report: dict):
    paths = report.get('paths', {})
    lengths = report.get('lengths_mm', {})
    mt = report.get('machining_time_estimate', {})
    print(f"[gcode] paths={paths.get('processed', 0)}/{paths.get('input', 0)} lines={report.get('total_gcode_lines', 0)} time={report.get('execution_time_ms', 0)}s")
    print(f"[gcode] cut={lengths.get('cutting', 0):.1f}mm rapid_xy={lengths.get('rapid_xy', 0):.1f}mm total_time={mt.get('total_time_formatted', 'N/A')}")
    warnings = report.get('warnings', [])
    if warnings:
        print(f"[gcode] warnings: {'; '.join(str(w) for w in warnings[:3])}")
