from typing import List, Tuple, Literal, Optional
import numpy as np

from engine.image_preprocessing import preprocess_pipeline, PreprocessResult
from engine.contour_extraction import (
    contour_extraction_pipeline,
    Contour,
    ExtractionReport,
)

def run_stage1(
    image_bgr: np.ndarray,
    wood_width_mm: float,
    wood_height_mm: float,
    tool_dia_mm: float,
    threshold_method: Literal["otsu", "adaptive"] = "otsu",
    use_clahe: bool = False,
    fit_mode: Literal["contain", "cover"] = "contain",
    classify_curves: bool = True,
    is_tiled_pattern: bool = False,
) -> Tuple[PreprocessResult, List[Contour], ExtractionReport]:
    result = preprocess_pipeline(
        image_bgr,
        wood_width_mm=wood_width_mm,
        wood_height_mm=wood_height_mm,
        tool_dia_mm=tool_dia_mm,
        threshold_method=threshold_method,
        use_clahe=use_clahe,
        fit_mode=fit_mode,
    )

    contours, report = contour_extraction_pipeline(
        result.binary,
        tool_dia_mm=tool_dia_mm,
        pixel_to_mm=result.pixel_to_mm,
        classify_curves=classify_curves,
        is_tiled_pattern=is_tiled_pattern,
    )

    return result, contours, report
