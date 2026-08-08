from typing import Tuple
import numpy as np

from engine.image_preprocessing import preprocess_pipeline, PreprocessResult


def run_stage1(
    image_bgr: np.ndarray,
    wood_width_mm: float,
    wood_height_mm: float,
    tool_dia_mm: float,
) -> PreprocessResult:
    result = preprocess_pipeline(
        image_bgr,
        wood_width_mm=wood_width_mm,
        wood_height_mm=wood_height_mm,
        tool_dia_mm=tool_dia_mm,
    )
    return result