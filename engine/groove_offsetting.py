"""Adaptive groove toolpath generation from the original binary ornament region.

This stage does not assume one fixed groove width.  The white foreground in the
preprocessed binary image is the exact region that must be machined.  Its local
width may vary from one part of the ornament to another.

Tool-centre paths are generated as inward distance contours:
    tool radius, tool radius + step-over, ...

Therefore every centre path stays at least one tool radius inside the original
boundary, so a cylindrical tool does not intentionally cut outside the source
ornament.  Wider regions automatically receive more passes; narrow regions
receive fewer passes.  Regions narrower than the tool diameter are reported as
unreachable with that tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
Path = List[Point]


@dataclass
class OffsetReport:
    mode: str
    tool_diameter_mm: float
    step_over_ratio: float
    pixel_to_mm: float
    distance_levels_mm: List[float] = field(default_factory=list)
    output_paths: int = 0
    collapsed_levels: int = 0
    foreground_area_mm2: float = 0.0
    machinable_area_mm2: float = 0.0
    unreachable_area_mm2: float = 0.0
    coverage_ratio_percent: float = 0.0
    minimum_detected_width_mm: float = 0.0
    maximum_detected_width_mm: float = 0.0
    notes: List[str] = field(default_factory=list)


def _validate_inputs(
    binary: np.ndarray,
    pixel_to_mm: float,
    tool_diameter_mm: float,
    step_over_ratio: float,
) -> None:
    if binary is None or binary.ndim != 2:
        raise ValueError("binary must be a single-channel image")
    if pixel_to_mm <= 0:
        raise ValueError("pixel_to_mm must be greater than zero")
    if tool_diameter_mm <= 0:
        raise ValueError("tool_diameter_mm must be greater than zero")
    if not (0 < step_over_ratio <= 1.0):
        raise ValueError("step_over_ratio must be in the range (0, 1]")


def _normalise_foreground(binary: np.ndarray) -> np.ndarray:
    """Return a clean uint8 mask where ornament material is 255."""
    mask = np.where(binary > 0, 255, 0).astype(np.uint8)

    # If almost the whole image is foreground, polarity is likely reversed.
    foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if foreground_ratio > 0.70:
        mask = cv2.bitwise_not(mask)

    return mask


def _ensure_closed(path: Sequence[Point]) -> Path:
    if len(path) < 3:
        return []
    result = [(float(x), float(y)) for x, y in path]
    if result[0] != result[-1]:
        result.append(result[0])
    return result


def _contours_from_level_mask(
    level_mask: np.ndarray,
    pixel_to_mm: float,
    min_path_length_mm: float,
) -> List[Path]:
    contours, _ = cv2.findContours(
        level_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    paths: List[Path] = []
    min_length_px = min_path_length_mm / pixel_to_mm

    for contour in contours:
        if contour is None or len(contour) < 3:
            continue

        perimeter_px = cv2.arcLength(contour, True)
        if perimeter_px < min_length_px:
            continue

        points = np.squeeze(contour, axis=1)
        path = _ensure_closed(
            [
                (float(point[0]) * pixel_to_mm,
                 float(point[1]) * pixel_to_mm)
                for point in points
            ]
        )
        if len(path) >= 4:
            paths.append(path)

    return paths


def _build_distance_levels(
    tool_radius_mm: float,
    maximum_distance_mm: float,
    max_step_mm: float,
    tolerance_mm: float = 1e-6,
) -> List[float]:
    if maximum_distance_mm + tolerance_mm < tool_radius_mm:
        return []

    levels = [tool_radius_mm]
    current = tool_radius_mm

    while current + max_step_mm < maximum_distance_mm - tolerance_mm:
        current += max_step_mm
        levels.append(current)

    # Add a final centre contour near the deepest part when the remaining gap
    # is large enough to matter. This covers wide cores without forcing every
    # region to use the same number of passes.
    if maximum_distance_mm - levels[-1] > max_step_mm * 0.35:
        levels.append(maximum_distance_mm)

    return levels


def generate_groove_offset_paths(
    binary: np.ndarray,
    pixel_to_mm: float,
    tool_diameter_mm: float,
    step_over_ratio: float = 0.60,
    min_path_length_mm: float = 0.50,
) -> Tuple[List[Path], OffsetReport]:
    """Generate adaptive toolpaths that follow the true widths in the image.

    Parameters
    ----------
    binary:
        Preprocessed binary image. White pixels represent the ornament/groove
        region to be machined.
    pixel_to_mm:
        Scale computed by the previous stage.
    tool_diameter_mm:
        Diameter of the cylindrical cutter.
    step_over_ratio:
        Maximum centre-to-centre spacing as a fraction of tool diameter.

    Returns
    -------
    paths, report
        Closed tool-centre paths in millimetres and a diagnostic report.
    """
    _validate_inputs(binary, pixel_to_mm, tool_diameter_mm, step_over_ratio)

    mask = _normalise_foreground(binary)
    foreground_pixels = int(np.count_nonzero(mask))
    pixel_area_mm2 = pixel_to_mm * pixel_to_mm

    distance_px = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    distance_mm = distance_px * pixel_to_mm

    tool_radius_mm = tool_diameter_mm / 2.0
    max_step_mm = tool_diameter_mm * step_over_ratio
    maximum_distance_mm = float(distance_mm.max())

    levels = _build_distance_levels(
        tool_radius_mm=tool_radius_mm,
        maximum_distance_mm=maximum_distance_mm,
        max_step_mm=max_step_mm,
    )

    report = OffsetReport(
        mode="ADAPTIVE_TRUE_WIDTH",
        tool_diameter_mm=tool_diameter_mm,
        step_over_ratio=step_over_ratio,
        pixel_to_mm=pixel_to_mm,
        distance_levels_mm=levels,
        foreground_area_mm2=foreground_pixels * pixel_area_mm2,
        machinable_area_mm2=0.0,
        unreachable_area_mm2=0.0,
        coverage_ratio_percent=0.0,
        minimum_detected_width_mm=(
            2.0 * float(distance_mm[distance_mm > 0].min())
            if np.any(distance_mm > 0) else 0.0
        ),
        maximum_detected_width_mm=2.0 * maximum_distance_mm,
    )

    output: List[Path] = []
    for level_mm in levels:
        # This iso-distance boundary is a valid tool-centre path: every point
        # lies level_mm inside the original ornament boundary.
        level_mask = np.where(distance_mm >= level_mm, 255, 0).astype(np.uint8)

        paths = _contours_from_level_mask(
            level_mask=level_mask,
            pixel_to_mm=pixel_to_mm,
            min_path_length_mm=min_path_length_mm,
        )

        if not paths:
            report.collapsed_levels += 1
            continue

        output.extend(paths)

    report.output_paths = len(output)

    # Raster coverage validation: draw every tool-centre path and expand it by
    # the physical tool radius.  This approximates the area actually swept by
    # the cutter and compares it with the original ornament region.
    centre_mask = np.zeros_like(mask)
    for path in output:
        points_px = np.array(
            [
                [int(round(x / pixel_to_mm)), int(round(y / pixel_to_mm))]
                for x, y in path
            ],
            dtype=np.int32,
        )
        if len(points_px) >= 2:
            cv2.polylines(centre_mask, [points_px], True, 255, 1, cv2.LINE_8)

    radius_px = max(1, int(round(tool_radius_mm / pixel_to_mm)))
    kernel_size = 2 * radius_px + 1
    tool_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    swept_mask = cv2.dilate(centre_mask, tool_kernel)
    covered_mask = cv2.bitwise_and(swept_mask, mask)

    covered_pixels = int(np.count_nonzero(covered_mask))
    unreachable_pixels = max(0, foreground_pixels - covered_pixels)
    report.machinable_area_mm2 = covered_pixels * pixel_area_mm2
    report.unreachable_area_mm2 = unreachable_pixels * pixel_area_mm2
    report.coverage_ratio_percent = (
        100.0 * covered_pixels / foreground_pixels
        if foreground_pixels > 0 else 0.0
    )

    if not levels:
        report.notes.append(
            "No part of the ornament is wide enough for the selected tool. "
            "Use a smaller cutter."
        )
    else:
        report.notes.append(
            "The groove width is not fixed. It is inherited directly from the "
            "white region in the preprocessed image."
        )
        report.notes.append(
            "Every generated centre path stays at least one tool radius inside "
            "the original ornament boundary, preventing intentional overcut."
        )
        report.notes.append(
            "Wide portions automatically receive more inward passes; narrow "
            "portions receive fewer passes."
        )

    if unreachable_pixels > 0:
        report.notes.append(
            f"{report.unreachable_area_mm2:.3f} mm² of very thin detail is "
            "narrower than the tool diameter and cannot be reached exactly."
        )

    return output, report


def print_offset_report(report: OffsetReport) -> None:
    levels_text = ", ".join(
        f"{value:.3f}" for value in report.distance_levels_mm
    )

    print(
        f"[offset] mode={report.mode} tool={report.tool_diameter_mm:.3f} mm "
        f"step_over={report.step_over_ratio:.0%}"
    )
    print(f"[offset] inward distance levels (mm): [{levels_text}]")
    print(
        f"[offset] detected local width range≈"
        f"{report.minimum_detected_width_mm:.3f}.."
        f"{report.maximum_detected_width_mm:.3f} mm"
    )
    print(
        f"[offset] output_paths={report.output_paths} "
        f"collapsed_levels={report.collapsed_levels}"
    )
    print(
        f"[offset] foreground_area={report.foreground_area_mm2:.3f} mm² "
        f"machinable={report.machinable_area_mm2:.3f} mm² "
        f"thin_unreachable={report.unreachable_area_mm2:.3f} mm² "
        f"machinable_ratio={report.coverage_ratio_percent:.2f}%"
    )

    for note in report.notes:
        print(f"[offset/note] {note}")
