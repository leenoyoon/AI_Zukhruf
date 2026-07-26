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

# -- مأخوذتين من الملف الأول (فرح): توليد G-code المتحقّق من مدخلات
#    المستخدم + معاينة المحاكاة HTML، بدل write_gcode البسيطة --
from engine.generate_Gcode import generate_gcode_from_user_input
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
):
    print(f"--- Processing: {os.path.basename(image_path)} ---")

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Previous stages remain unchanged.
    result, contours, report = run_stage1(
        image,
        wood_width_mm=wood_width_mm,
        wood_height_mm=wood_height_mm,
        tool_dia_mm=tool_dia_mm,
    )

    cv2.imwrite("check_binary.png", result.binary)

    print(
        f"[stage1] total={report.total_found} kept={report.kept} "
        f"dropped_small={report.dropped_too_small} "
        f"dropped_bg={report.dropped_as_background} "
        f"pixel_to_mm={result.pixel_to_mm:.4f}"
    )

    for note in result.scale_notes:
        print("[stage1/scale]", note)

    # -------- Adaptive offset stage (سارة) -- ما تغيّر ولا سطر هون --------
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
            offset_paths, new_report = regenerate_with_suggested_tool(
                result.binary, result.pixel_to_mm, advice
            )
            print_offset_report(new_report)
            used_tool_mm = advice.suggested_tool_mm

    if not offset_paths:
        print("[offset/error] No machinable paths were generated. Use a smaller tool.")
        return

    # -------- Simplify + Optimize (advanced: DP + GA + 2-Opt++) --------
    # NOTE: optimize_paths_advanced() simplifies internally (epsilon_mm=0.15),
    # exactly like the original pathOptimizstion.py script did -- pass it the
    # RAW offset_paths here, not an already-simplified list, or it gets
    # simplified twice.
    _final_route, ordered = optimize_paths_advanced(offset_paths)

    # -------- G-code + Simulation (فرح) -- هون التبديل --------
    user_settings = {
        "safe_z": safe_z,
        "cut_depth": depth,
        "step_down": step_down,
        "feed_rate": feed_rate,
        "plunge_rate": plunge_rate,
        "spindle_speed": spindle_speed,
    }
    gcode_content = generate_gcode_from_user_input(
        optimized_paths=ordered,
        user_settings=user_settings,
    )

    gcode_filename = os.path.basename(output_path)
    html_filename = gcode_filename.replace(".gcode", "_preview.html")
    html_simulation_path = os.path.join(Config.SIMULATION_DIR, html_filename)
    generate_gcode_simulation_html(gcode_content, html_simulation_path)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(gcode_content)

    print(f"[G-Code] Successfully generated and saved to: {output_path} "
          f"(tool actually used: {used_tool_mm}mm)")
    print(f"[Simulation] Preview saved to: {html_simulation_path}")


if __name__ == "__main__":
    input_image = os.path.join(Config.INPUT_DIR, "pattern2.jpg")
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