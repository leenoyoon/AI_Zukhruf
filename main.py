import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))

from config import Config
from engine.preprocessing_stage import run_stage1
from engine.groove_offsetting import (
    generate_groove_offset_paths,
    print_offset_report,
)
from engine.dphull_integration import simplify_offset_paths
from engine.gcode_generator import optimize_paths, write_gcode


def process_image_to_gcode(
    image_path,
    output_path,
    wood_width_mm=300.0,
    wood_height_mm=300.0,
    tool_dia_mm=2.0,
    step_over_ratio=0.60,
    depth=-3.0,
    step_down=1.0,
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

    # Adaptive offset stage: the true groove widths come directly from the
    # preprocessed binary ornament, so no fixed groove_width_mm is required.
    offset_paths, offset_report = generate_groove_offset_paths(
        binary=result.binary,
        pixel_to_mm=result.pixel_to_mm,
        tool_diameter_mm=tool_dia_mm,
        step_over_ratio=step_over_ratio,
    )
    print_offset_report(offset_report)

    if not offset_paths:
        print("[offset/error] No machinable paths were generated. Use a smaller tool.")
        return

    simplified = simplify_offset_paths(offset_paths, epsilon_mm=0.15)
    ordered = optimize_paths(simplified)
    write_gcode(
        ordered,
        output_path,
        depth=depth,
        step_down=step_down,
    )


if __name__ == "__main__":
    input_image = os.path.join(Config.INPUT_DIR, "pattern24.jpg")
    output_gcode = os.path.join(Config.OUTPUT_DIR, "final_zukhruf25.gcode")

    process_image_to_gcode(
        image_path=input_image,
        output_path=output_gcode,
        wood_width_mm=300.0,
        wood_height_mm=300.0,
        tool_dia_mm=2.0,
        step_over_ratio=0.60,
        depth=-3.0,
    )
