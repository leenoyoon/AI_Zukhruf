from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple
import cv2
import numpy as np
from mask_contour_extraction import (
    Point,
    Path,
    _contours_from_level_mask_subpixel,
)

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


def suggest_tool_for_full_coverage(
    binary,
    pixel_to_mm: float,
    current_tool_mm: float,
    coverage_threshold_percent: float = 99.0,
    standard_sizes_mm: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0),
    **generate_kwargs,
) -> "tuple[float, OffsetReport] | None":
    candidates = sorted(
        (s for s in standard_sizes_mm if s < current_tool_mm), reverse=True
    )
    for size in candidates:
        _, trial_report = generate_groove_offset_paths(
            binary, pixel_to_mm, size, **generate_kwargs
        )
        if trial_report.coverage_ratio_percent >= coverage_threshold_percent:
            return size, trial_report
    return None


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
    foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if foreground_ratio > 0.70:
        mask = cv2.bitwise_not(mask)

    return mask


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
    _validate_inputs(binary, pixel_to_mm, tool_diameter_mm, step_over_ratio)

    mask = _normalise_foreground(binary)
    foreground_pixels = int(np.count_nonzero(mask))
    pixel_area_mm2 = pixel_to_mm * pixel_to_mm

    distance_px = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
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
        paths = _contours_from_level_mask_subpixel(
            distance_mm=distance_mm,
            level_mm=level_mm,
            pixel_to_mm=pixel_to_mm,
            min_path_length_mm=min_path_length_mm,
        )
        if not paths:
            report.collapsed_levels += 1
            continue

        output.extend(paths)

    report.output_paths = len(output)
    
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