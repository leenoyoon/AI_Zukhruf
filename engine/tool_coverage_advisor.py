from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from groove_offsetting import OffsetReport, Path, estimate_coverage_only, generate_groove_offset_paths, suggest_tool_for_full_coverage, suggest_tool_for_full_coverage_fast, suggest_tool_for_full_coverage_hybrid

@dataclass
class ToolAdvice:
    coverage_ok: bool
    report: OffsetReport
    paths_with_chosen_tool: List[Path]
    chosen_tool_mm: float
    suggested_tool_mm: Optional[float]
    suggested_tool_report: Optional[OffsetReport]
    message: str

def generate_with_coverage_advice(binary: np.ndarray, pixel_to_mm: float, tool_diameter_mm: float, coverage_threshold_percent: float=99.0, **kwargs) -> ToolAdvice:
    paths, report = generate_groove_offset_paths(binary, pixel_to_mm, tool_diameter_mm, **kwargs)
    if report.coverage_ratio_percent >= coverage_threshold_percent:
        return ToolAdvice(coverage_ok=True, report=report, paths_with_chosen_tool=paths, chosen_tool_mm=tool_diameter_mm, suggested_tool_mm=None, suggested_tool_report=None, message=f'Coverage {report.coverage_ratio_percent:.1f}% -- OK, proceed as usual.')
    suggestion = suggest_tool_for_full_coverage(binary, pixel_to_mm, tool_diameter_mm)
    suggested = suggestion[0] if suggestion else None
    suggested_report = suggestion[1] if suggestion else None
    msg = f'This tool ({tool_diameter_mm}mm) only covers {report.coverage_ratio_percent:.1f}% ({report.unreachable_area_mm2:.2f}mm² will not be carved). '
    msg += f'Switch to a {suggested}mm tool for near-full coverage, or continue with the current tool.' if suggested is not None else 'No standard tool smaller than the current one in our list fully covers this detail -- the design itself may need adjustment, or add a smaller tool size to the list.'
    return ToolAdvice(coverage_ok=False, report=report, paths_with_chosen_tool=paths, chosen_tool_mm=tool_diameter_mm, suggested_tool_mm=suggested, suggested_tool_report=suggested_report, message=msg)

def generate_with_coverage_advice_preflight(binary: np.ndarray, pixel_to_mm: float, tool_diameter_mm: float, coverage_threshold_percent: float=99.0, step_over_ratio: float=0.6) -> ToolAdvice:
    _, report = generate_groove_offset_paths(binary, pixel_to_mm, tool_diameter_mm, step_over_ratio=step_over_ratio)
    if report.coverage_ratio_percent >= coverage_threshold_percent:
        return ToolAdvice(coverage_ok=True, report=report, paths_with_chosen_tool=[], chosen_tool_mm=tool_diameter_mm, suggested_tool_mm=None, suggested_tool_report=None, message=f'Coverage {report.coverage_ratio_percent:.1f}% -- OK, proceed as usual.')
    suggestion = suggest_tool_for_full_coverage_hybrid(binary, pixel_to_mm, tool_diameter_mm, coverage_threshold_percent=coverage_threshold_percent, step_over_ratio=step_over_ratio)
    suggested = suggestion[0] if suggestion else None
    suggested_report = suggestion[1] if suggestion else None
    msg = f'This tool ({tool_diameter_mm}mm) only covers {report.coverage_ratio_percent:.1f}% ({report.unreachable_area_mm2:.2f}mm² will not be carved). '
    msg += f'Switch to a {suggested}mm tool for near-full coverage, or continue with the current tool.' if suggested is not None else 'No standard tool smaller than the current one in our list fully covers this detail -- the design itself may need adjustment, or add a smaller tool size to the list.'
    return ToolAdvice(coverage_ok=False, report=report, paths_with_chosen_tool=[], chosen_tool_mm=tool_diameter_mm, suggested_tool_mm=suggested, suggested_tool_report=suggested_report, message=msg)

def generate_with_coverage_advice_fast(binary: np.ndarray, pixel_to_mm: float, tool_diameter_mm: float, coverage_threshold_percent: float=99.0, step_over_ratio: float=0.6) -> ToolAdvice:
    report = estimate_coverage_only(binary, pixel_to_mm, tool_diameter_mm, step_over_ratio=step_over_ratio)
    if report.coverage_ratio_percent >= coverage_threshold_percent:
        return ToolAdvice(coverage_ok=True, report=report, paths_with_chosen_tool=[], chosen_tool_mm=tool_diameter_mm, suggested_tool_mm=None, suggested_tool_report=None, message=f'Coverage {report.coverage_ratio_percent:.1f}% -- OK, proceed as usual.')
    suggestion = suggest_tool_for_full_coverage_fast(binary, pixel_to_mm, tool_diameter_mm, coverage_threshold_percent=coverage_threshold_percent, step_over_ratio=step_over_ratio)
    suggested = suggestion[0] if suggestion else None
    suggested_report = suggestion[1] if suggestion else None
    msg = f'This tool ({tool_diameter_mm}mm) only covers {report.coverage_ratio_percent:.1f}% ({report.unreachable_area_mm2:.2f}mm² will not be carved). '
    msg += f'Switch to a {suggested}mm tool for near-full coverage, or continue with the current tool.' if suggested is not None else 'No standard tool smaller than the current one in our list fully covers this detail -- the design itself may need adjustment, or add a smaller tool size to the list.'
    return ToolAdvice(coverage_ok=False, report=report, paths_with_chosen_tool=[], chosen_tool_mm=tool_diameter_mm, suggested_tool_mm=suggested, suggested_tool_report=suggested_report, message=msg)

def regenerate_with_suggested_tool(binary: np.ndarray, pixel_to_mm: float, advice: ToolAdvice) -> Tuple[List[Path], OffsetReport]:
    if advice.suggested_tool_mm is None or advice.suggested_tool_report is None:
        raise ValueError('No suggested tool for this case')
    return generate_groove_offset_paths(binary, pixel_to_mm, advice.suggested_tool_mm)
