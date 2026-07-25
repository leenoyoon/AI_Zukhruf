"""
tool_coverage_advisor.py
--------------------------------------------------------------------------
طبقة تنسيق رفيعة فوق groove_offsetting.py -- لا تُعدّل خوارزمية سارة
إطلاقاً، بس بتقرر (بعد التوليد الأول، ببلاش، من نفس التقرير) هل نحتاج
نسأل المستخدم قرار ولا لأ.

التدفق (خيارين بس، متل ما تقرر بالمحادثة -- بلا ملفات متعددة):
  1) توليد بالأداة المختارة (توليد واحد فعلي -- مو تجربة/تجاهل).
  2) لو التغطية < العتبة: رجّع ToolAdvice فيها المسارات الجاهزة أصلاً
     (خيار أ) + قطر أداة مقترح (خيار ب)، بلا ما تولّدي فيها فوراً.
  3) لو المستخدم اختار (ب): استدعاء واحد إضافي بالأداة المقترحة --
     وقتها بس، مو قبل.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from groove_offsetting import (
    OffsetReport,
    Path,
    generate_groove_offset_paths,
    suggest_tool_for_full_coverage,
)


@dataclass
class ToolAdvice:
    coverage_ok: bool
    report: OffsetReport
    paths_with_chosen_tool: List[Path]
    chosen_tool_mm: float
    suggested_tool_mm: Optional[float]
    suggested_tool_report: Optional[OffsetReport]
    message: str


def generate_with_coverage_advice(
    binary: np.ndarray,
    pixel_to_mm: float,
    tool_diameter_mm: float,
    coverage_threshold_percent: float = 99.0,
    **kwargs,
) -> ToolAdvice:
    paths, report = generate_groove_offset_paths(
        binary, pixel_to_mm, tool_diameter_mm, **kwargs
    )

    if report.coverage_ratio_percent >= coverage_threshold_percent:
        return ToolAdvice(
            coverage_ok=True,
            report=report,
            paths_with_chosen_tool=paths,
            chosen_tool_mm=tool_diameter_mm,
            suggested_tool_mm=None,
            suggested_tool_report=None,
            message=f"Coverage {report.coverage_ratio_percent:.1f}% -- OK, proceed as usual.",
        )

    suggestion = suggest_tool_for_full_coverage(binary, pixel_to_mm, tool_diameter_mm)
    suggested = suggestion[0] if suggestion else None
    suggested_report = suggestion[1] if suggestion else None

    msg = (
        f"This tool ({tool_diameter_mm}mm) only covers {report.coverage_ratio_percent:.1f}% "
        f"({report.unreachable_area_mm2:.2f}mm² will not be carved). "
    )
    msg += (
        f"Switch to a {suggested}mm tool for near-full coverage, or continue with the current tool."
        if suggested is not None else
        "No standard tool smaller than the current one in our list fully covers this detail -- "
        "the design itself may need adjustment, or add a smaller tool size to the list."
    )
    return ToolAdvice(
        coverage_ok=False,
        report=report,
        paths_with_chosen_tool=paths,   # جاهزة فوراً لو المستخدم اختار "كمّلي بنفس الأداة"
        chosen_tool_mm=tool_diameter_mm,
        suggested_tool_mm=suggested,
        suggested_tool_report=suggested_report,
        message=msg,
    )


def regenerate_with_suggested_tool(
    binary: np.ndarray,
    pixel_to_mm: float,
    advice: ToolAdvice,
) -> Tuple[List[Path], OffsetReport]:
    """
    يتنادى فقط لو المستخدم اختار صراحة 'بدّلي الأداة' (خيار ب).
    ملاحظة: النتيجة محسوبة أصلاً جوا suggest_tool_for_full_coverage
    (كجزء من البحث)، فمو محتاجين نعيد التوليد من الصفر -- بس نرجعها.
    """
    if advice.suggested_tool_mm is None or advice.suggested_tool_report is None:
        raise ValueError("ما في أداة مقترحة لهالحالة")
    # المسارات نفسها ما انخزنت بالبحث (وفراً للذاكرة)، فبس هلق منولدها
    # مرة وحيدة فعلية بالقطر المقترح.
    return generate_groove_offset_paths(binary, pixel_to_mm, advice.suggested_tool_mm)
