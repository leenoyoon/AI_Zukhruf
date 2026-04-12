import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, "data", "input_images")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output_gcode")
    TEXTURE_DIR = os.path.join(BASE_DIR, "data", "generated_textures")

    IMAGE_SIZE = (512, 512)
    DEFAULT_PATCH_SIZE = 64
    DEFAULT_OVERLAP_RATIO = 0.25
