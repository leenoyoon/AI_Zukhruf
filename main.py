import os
import sys
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))
from config import Config
from engine.preprocessing_stage import run_stage1
from engine.groove_offsetting import print_offset_report
from engine.tool_coverage_advisor import (
    generate_with_coverage_advice,
    regenerate_with_suggested_tool,
)
from engine.pathOptimizstion import optimize_paths_advanced
from engine.generate_Gcode import generate_gcode_from_user_input_with_report, print_gcode_report
from engine.simulate import generate_gcode_simulation_html

def process_image_to_gcode(
    image_path,
    output_path,
    wood_width_mm=300.0,
    wood_height_mm=300.0,
    tool_dia_mm=2.0,
    step_over_ratio=0.60,
    depth=-3.0,
    step_down=1.0,
    feed_rate=800.0,
    plunge_rate=300.0,
    spindle_speed=12000,
    safe_z=5.0,
    machine_hourly_rate=20,  
):
    print(f"--- Processing: {os.path.basename(image_path)} ---")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return
    
    result = run_stage1(
        image,
        wood_width_mm=wood_width_mm,
        wood_height_mm=wood_height_mm,
        tool_dia_mm=tool_dia_mm,
    )

    # ===== DEBUG CENTERING =====
    print(f"[centering] offset = ({result.offset_x_mm:.2f}, {result.offset_y_mm:.2f})")
    print(f"[centering] pixel_to_mm = {result.pixel_to_mm:.4f}")
    print(f"[centering] pad_px = {result.pad_px}")
    # ===========================

    cv2.imwrite("check_binary.png", result.binary)
    print(f"[stage1] pixel_to_mm={result.pixel_to_mm:.4f}")

    for note in result.scale_notes:
        print("[stage1/scale]", note)

    
    advice = generate_with_coverage_advice(
        binary=result.binary,
        pixel_to_mm=result.pixel_to_mm,
        tool_diameter_mm=tool_dia_mm,
        step_over_ratio=step_over_ratio,
    )
    print_offset_report(advice.report)
    print(f"[coverage] {advice.message}")

    offset_paths = advice.paths_with_chosen_tool
    used_tool_mm = tool_dia_mm

    if not advice.coverage_ok and advice.suggested_tool_mm is not None:
        answer = input(
            f"Switch to the suggested tool diameter {advice.suggested_tool_mm}mm "
            f"for better coverage? [y/N]: "
        ).strip().lower()
        if answer.startswith("y"):
            used_tool_mm = advice.suggested_tool_mm
            print(f"[coverage] Re-preprocessing with tool {used_tool_mm}mm ...")

            result = run_stage1(
                image,
                wood_width_mm=wood_width_mm,
                wood_height_mm=wood_height_mm,
                tool_dia_mm=used_tool_mm,
            )
            cv2.imwrite("check_binary.png", result.binary)
            print(f"[stage1] pixel_to_mm={result.pixel_to_mm:.4f} (after tool switch)")

            from engine.groove_offsetting import generate_groove_offset_paths
            offset_paths, new_report = generate_groove_offset_paths(
                result.binary,
                result.pixel_to_mm,
                used_tool_mm,
                step_over_ratio=step_over_ratio,
            )
            print_offset_report(new_report)

    if not offset_paths:
        print("[offset/error] No machinable paths were generated. Use a smaller tool.")
        return
    
    # -- توسيط النقشة
    if result.offset_x_mm or result.offset_y_mm:
        print(f"[centering] shifting design by "
              f"({result.offset_x_mm:.2f}, {result.offset_y_mm:.2f}) mm "
              f"to center it on the {wood_width_mm}x{wood_height_mm} mm stock")
        offset_paths = [
            [(x + result.offset_x_mm, y + result.offset_y_mm) for x, y in path]
            for path in offset_paths
        ]

    # ===== DEBUG بعد الإزاحة =====
    all_x = [p[0] for path in offset_paths for p in path]
    all_y = [p[1] for path in offset_paths for p in path]
    print(f"[centering] after shift → X range: {min(all_x):.1f} .. {max(all_x):.1f}")
    print(f"[centering] after shift → Y range: {min(all_y):.1f} .. {max(all_y):.1f}")
    # ==============================
    
    _final_route, ordered = optimize_paths_advanced(
        offset_paths,
        pixel_to_mm=result.pixel_to_mm,
    )

    user_settings = {
        "safe_z": safe_z,
        "cut_depth": depth,
        "step_down": step_down,
        "feed_rate": feed_rate,
        "plunge_rate": plunge_rate,
        "spindle_speed": spindle_speed,
        "tool_diameter_mm": used_tool_mm,
        "stepover_mm": used_tool_mm * step_over_ratio,
        "machine_hourly_rate": machine_hourly_rate,
    }
    gcode_content, report = generate_gcode_from_user_input_with_report(
        optimized_paths=ordered,
        user_settings=user_settings,
    )

    gcode_filename = os.path.basename(output_path)
    html_filename = gcode_filename.replace(".gcode", "_preview.html")
    html_simulation_path = os.path.join(Config.SIMULATION_DIR, html_filename)
    generate_gcode_simulation_html(
        gcode_content,
        html_simulation_path,
        wood_width_mm=wood_width_mm,
        wood_height_mm=wood_height_mm,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(gcode_content)

    print(f"[G-Code] Successfully generated and saved to: {output_path} "
          f"(tool actually used: {used_tool_mm}mm)")
    print(f"[Simulation] Preview saved to: {html_simulation_path}")

    print_gcode_report(report)


if __name__ == "__main__":
    input_image = os.path.join(Config.INPUT_DIR, "pattern16.jpg")
    output_gcode = os.path.join(Config.OUTPUT_DIR, "final_zukhruf25.gcode")

    process_image_to_gcode(
        image_path=input_image,
        output_path=output_gcode,
        wood_width_mm=300.0,
        wood_height_mm=300.0,
        tool_dia_mm=2.0,
        step_over_ratio=0.60,
        depth=-3.0,
        step_down=1.0,
        feed_rate=800.0,
        plunge_rate=300.0,
        spindle_speed=12000,
        safe_z=5.0,
    )