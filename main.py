import os
import sys
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))
from config import Config
from engine.preprocessing_stage import run_stage1
from engine.pathOffset import offset_contours
from engine.dphull_integration import simplify_offset_paths
from engine.gcode_generator import optimize_paths, write_gcode

def process_image_to_gcode(
    image_path,
    output_path,
    wood_width_mm=300.0,
    wood_height_mm=300.0,
    tool_dia_mm=2.0,
    depth=-3.0,
    step_down=1.0,
):
    print(f"--- Processing: {os.path.basename(image_path)} ---")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found!")
        return

    
    result, contours, report = run_stage1(
        img, wood_width_mm=wood_width_mm, wood_height_mm=wood_height_mm,
        tool_dia_mm=tool_dia_mm,
    )
    cv2.imwrite("check_binary.png", result.binary)
    print(f"[stage1] total={report.total_found} kept={report.kept} "
          f"dropped_small={report.dropped_too_small} "
          f"dropped_bg={report.dropped_as_background} "
          f"pixel_to_mm={result.pixel_to_mm:.4f}")
    for n in result.scale_notes:
        print("[stage1/scale]", n)

    
    offset_paths = offset_contours(
        contours, pixel_to_mm=result.pixel_to_mm, tool_dia_mm=tool_dia_mm,
    )

    
    simplified = simplify_offset_paths(offset_paths, epsilon_mm=0.15)

    
    
    ordered = optimize_paths(simplified)
    write_gcode(ordered, output_path, depth=depth, step_down=step_down)


if __name__ == "__main__":
    input_image = os.path.join(Config.INPUT_DIR, "pattern18.jpg")
    output_gcode = os.path.join(Config.OUTPUT_DIR, "final_zukhruf25.gcode")
    process_image_to_gcode(
        image_path=input_image,
        output_path=output_gcode,
        wood_width_mm=300.0,
        wood_height_mm=300.0,
        tool_dia_mm=2.0,
        depth=-3.0,
    )
