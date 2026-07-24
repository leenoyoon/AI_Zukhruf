import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "engine",
    ),
)

import cv2
import matplotlib.pyplot as plt

from config import Config
from engine.preprocessing_stage import run_stage1
from engine.groove_offsetting import (
    generate_groove_offset_paths,
    print_offset_report,
)


def _split_xy(path):
    """
    يفصل المسار إلى X و Y للرسم.
    """
    return (
        [float(point[0]) for point in path],
        [float(point[1]) for point in path],
    )


def _contour_to_mm(contour, pixel_to_mm):
    """
    يحوّل نقاط الكونتور من Pixel إلى mm ويغلقه بصرياً.
    """
    points = getattr(contour, "points", None)

    if points is None or len(points) < 2:
        return None

    path = [
        (
            float(point[0]) * pixel_to_mm,
            float(point[1]) * pixel_to_mm,
        )
        for point in points
    ]

    if path[0] != path[-1]:
        path.append(path[0])

    return path


def _binary_extent_mm(binary, pixel_to_mm):
    """
    حدود عرض الصورة بالـ mm.
    """
    height, width = binary.shape[:2]

    return [
        0,
        width * pixel_to_mm,
        height * pixel_to_mm,
        0,
    ]


def _apply_same_axes(ax1, ax2, extent):
    """
    يوحّد مجال الرسم بين الشكلين حتى تكون المقارنة صحيحة بصرياً.
    """
    x_min, x_max, y_max, y_min = extent

    for axis in (ax1, ax2):
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_max, y_min)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("X (mm)")
        axis.set_ylabel("Y (mm)")
        axis.grid(True)


def plot_contours_and_offset(
    contours,
    offset_paths,
    binary,
    pixel_to_mm,
    tool_dia_mm,
):
    """
    النافذة الأولى:

    اليسار: الكونتورات الأصلية فقط.
    اليمين: مسارات الأوفست النهائية فقط.

    يتم استخدام نفس المقياس ونفس حدود المحاور في الشكلين،
    حتى تظهر المقارنة بالحجم الحقيقي دون تكبير أو تصغير مختلف.
    """
    extent = _binary_extent_mm(
        binary=binary,
        pixel_to_mm=pixel_to_mm,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
    )

    contour_ax = axes[0]
    offset_ax = axes[1]

    # ---------------------------------------------------------
    # اليسار: الكونتورات الأصلية
    # ---------------------------------------------------------
    for contour in contours:
        contour_path = _contour_to_mm(
            contour=contour,
            pixel_to_mm=pixel_to_mm,
        )

        if contour_path is None:
            continue

        xs, ys = _split_xy(contour_path)

        contour_ax.plot(
            xs,
            ys,
            color="lime",
            linewidth=1.3,
        )

    contour_ax.set_title(
        f"Original Extracted Contours\n"
        f"Contours = {len(contours)}"
    )

    # ---------------------------------------------------------
    # اليمين: مسارات مركز الأداة الناتجة
    # ---------------------------------------------------------
    for path in offset_paths:
        if path is None or len(path) < 2:
            continue

        xs, ys = _split_xy(path)

        offset_ax.plot(
            xs,
            ys,
            color="red",
            linewidth=1.0,
        )

    offset_ax.set_title(
        f"Final Adaptive Offset Paths\n"
        f"Tool = {tool_dia_mm:.1f} mm | "
        f"Paths = {len(offset_paths)}"
    )

    _apply_same_axes(
        contour_ax,
        offset_ax,
        extent,
    )

    figure.suptitle(
        "Contour vs Final Adaptive Offset",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


def plot_overlay_and_coverage(
    contours,
    offset_paths,
    binary,
    pixel_to_mm,
    tool_dia_mm,
):
    """
    النافذة الثانية:

    اليسار: الكونتور الأخضر مع مسارات الأوفست الحمراء.
    اليمين: محاكاة تقريبية للمنطقة التي يغطيها قطر الأداة.
    """
    extent = _binary_extent_mm(
        binary=binary,
        pixel_to_mm=pixel_to_mm,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
    )

    overlay_ax = axes[0]
    coverage_ax = axes[1]

    # ---------------------------------------------------------
    # اليسار: Overlay
    # ---------------------------------------------------------
    for contour in contours:
        contour_path = _contour_to_mm(
            contour=contour,
            pixel_to_mm=pixel_to_mm,
        )

        if contour_path is None:
            continue

        xs, ys = _split_xy(contour_path)

        overlay_ax.plot(
            xs,
            ys,
            color="lime",
            linewidth=1.4,
            label=None,
        )

    for path in offset_paths:
        if path is None or len(path) < 2:
            continue

        xs, ys = _split_xy(path)

        overlay_ax.plot(
            xs,
            ys,
            color="red",
            linewidth=0.9,
            alpha=0.90,
        )

    overlay_ax.set_title(
        "Overlay\n"
        "Green = Original contours | "
        "Red = Final offset paths"
    )

    # ---------------------------------------------------------
    # اليمين: المنطقة الأصلية + عرض الأداة الحقيقي تقريبياً
    # ---------------------------------------------------------
    coverage_ax.imshow(
        binary,
        cmap="gray",
        extent=extent,
        interpolation="nearest",
        alpha=0.35,
    )

    # Matplotlib linewidth is measured in points.
    tool_linewidth_points = (
        tool_dia_mm * 72.0 / 25.4
    )

    for path in offset_paths:
        if path is None or len(path) < 2:
            continue

        xs, ys = _split_xy(path)

        coverage_ax.plot(
            xs,
            ys,
            color="red",
            linewidth=tool_linewidth_points,
            solid_capstyle="round",
            solid_joinstyle="round",
            alpha=0.55,
        )

    coverage_ax.set_title(
        "Approximate Swept Tool Area\n"
        f"Displayed stroke width = "
        f"{tool_dia_mm:.1f} mm tool diameter"
    )

    _apply_same_axes(
        overlay_ax,
        coverage_ax,
        extent,
    )

    figure.suptitle(
        "Adaptive Offset — Overlay and Coverage",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


def test_offset_stage(
    image_path,
    wood_width_mm=300.0,
    wood_height_mm=300.0,
    tool_dia_mm=2.0,
    step_over_ratio=0.60,
):
    """
    يشغّل فقط:

    Preprocessing
    → Contour Extraction
    → Adaptive Groove Offsetting
    → Visualization

    ولا يشغّل:
    DPHull
    Path Optimization
    G-code Generation
    """
    print(
        f"--- Adaptive offset test: "
        f"{os.path.basename(image_path)} ---"
    )

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    result, contours, extraction_report = run_stage1(
        image_bgr=image,
        wood_width_mm=wood_width_mm,
        wood_height_mm=wood_height_mm,
        tool_dia_mm=tool_dia_mm,
    )

    print(
        f"[stage1] total={extraction_report.total_found} "
        f"kept={extraction_report.kept} "
        f"pixel_to_mm={result.pixel_to_mm:.4f}"
    )

    offset_paths, offset_report = (
        generate_groove_offset_paths(
            binary=result.binary,
            pixel_to_mm=result.pixel_to_mm,
            tool_diameter_mm=tool_dia_mm,
            step_over_ratio=step_over_ratio,
        )
    )

    print_offset_report(offset_report)

    # النافذة الأولى:
    # الكونتور الأصلي مقابل الأوفست النهائي.
    plot_contours_and_offset(
        contours=contours,
        offset_paths=offset_paths,
        binary=result.binary,
        pixel_to_mm=result.pixel_to_mm,
        tool_dia_mm=tool_dia_mm,
    )

    # بعد إغلاق النافذة الأولى تظهر النافذة الثانية.
    plot_overlay_and_coverage(
        contours=contours,
        offset_paths=offset_paths,
        binary=result.binary,
        pixel_to_mm=result.pixel_to_mm,
        tool_dia_mm=tool_dia_mm,
    )


if __name__ == "__main__":
    input_image = os.path.join(
        Config.INPUT_DIR,
        "pattern2.jpg",
    )

    test_offset_stage(
        image_path=input_image,
        wood_width_mm=300.0,
        wood_height_mm=300.0,
        tool_dia_mm=1.0,
        step_over_ratio=0.60,
    )
