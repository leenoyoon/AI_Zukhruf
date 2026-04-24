import os
from config import Config
from engine.gcode_generator import process_image_to_gcode

if __name__ == "__main__":
    input_image = os.path.join(Config.INPUT_DIR, "pattern18.jpg")
    output_gcode = os.path.join(Config.OUTPUT_DIR, "final_zukhruf2.gcode")
    process_image_to_gcode(
        image_path=input_image,
        output_path=output_gcode,
        pixel_to_mm=0.5,   
        tool_dia=2.0,      
        depth=-3.0         
    )