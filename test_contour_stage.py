import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# تجهيز المسارات
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENGINE_DIR = BASE_DIR / "engine"

# مهم لأن بعض ملفات المشروع تستخدم imports بدون engine.
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))


# ============================================================
# استيراد المراحل الثلاث من المشروع
# ============================================================

from image_preprocessing import preprocess_pipeline
from contour_extraction import contour_extraction_pipeline

from contour_pipeline import (
    Contour as SimplificationContour,
    simplify_pipeline,
)


# ============================================================
# الإعدادات
# ============================================================

IMAGE_PATH = BASE_DIR / "data" / "input_images" / "pattern18.jpg"

OUTPUT_DIR = BASE_DIR / "data" / "output_images"

WOOD_WIDTH_MM = 300.0
WOOD_HEIGHT_MM = 300.0
TOOL_DIAMETER_MM = 2.0

# مقدار تبسيط DPHull
# كلما زادت القيمة، تنحذف نقاط أكثر.
DPHULL_EPSILON_MM = 0.7

# أقل مسافة مسموحة بين نقطتين
MIN_SEGMENT_MM = 0.02


# ============================================================
# دوال مساعدة
# ============================================================

def resize_for_display(image, max_width=850, max_height=700):
    """
    تصغير الصورة للعرض فقط، من دون التأثير على الحسابات أو الصور المحفوظة.
    """

    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale >= 1.0:
        return image

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def draw_contours_on_binary(binary_image, contours, thickness=2):
    """
    رسم الكونتورات على نسخة ملونة من الصورة الثنائية.

    الأحمر: الكونتور الخارجي
    الأخضر: الثقوب
    """

    output = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        points = np.asarray(contour.points, dtype=np.int32)

        if len(points) < 3:
            continue

        points = points.reshape((-1, 1, 2))

        if contour.is_hole:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.drawContours(
            image=output,
            contours=[points],
            contourIdx=-1,
            color=color,
            thickness=thickness,
        )

    return output


def draw_before_after_comparison(
    binary_image,
    original_contours,
    simplified_contours,
):
    """
    رسم الكونتور قبل وبعد DPHull فوق نفس الصورة.

    الأزرق: قبل التبسيط
    الأحمر: بعد التبسيط
    """

    comparison = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # قبل DPHull باللون الأزرق
    for contour in original_contours:
        points = np.asarray(contour.points, dtype=np.int32)

        if len(points) < 3:
            continue

        points = points.reshape((-1, 1, 2))

        cv2.drawContours(
            comparison,
            [points],
            -1,
            (255, 0, 0),
            2,
        )

    # بعد DPHull باللون الأحمر
    for contour in simplified_contours:
        points = np.asarray(contour.points, dtype=np.int32)

        if len(points) < 3:
            continue

        points = points.reshape((-1, 1, 2))

        cv2.drawContours(
            comparison,
            [points],
            -1,
            (0, 0, 255),
            1,
        )

        # إظهار نقاط DPHull نفسها
        for x, y in contour.points:
            cv2.circle(
                comparison,
                (int(round(x)), int(round(y))),
                radius=3,
                color=(0, 255, 255),
                thickness=-1,
            )

    return comparison


# ============================================================
# البرنامج الأساسي
# ============================================================

def main():
    print("=" * 60)
    print("Preprocessing + Contour Extraction + DPHull Test")
    print("=" * 60)

    print(f"\nProject directory:\n{BASE_DIR}")
    print(f"\nImage path:\n{IMAGE_PATH}")

    # --------------------------------------------------------
    # التأكد من وجود الصورة
    # --------------------------------------------------------

    if not IMAGE_PATH.exists():
        input_folder = BASE_DIR / "data" / "input_images"

        existing_images = []

        if input_folder.exists():
            existing_images = [
                file.name
                for file in input_folder.iterdir()
                if file.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            ]

        message = [
            "",
            "الصورة غير موجودة في المسار:",
            str(IMAGE_PATH),
        ]

        if existing_images:
            message.append("")
            message.append("الصور الموجودة داخل input_images:")
            message.extend(f"- {name}" for name in existing_images)
        else:
            message.append("")
            message.append(
                "مجلد data/input_images غير موجود أو لا يحتوي صورًا."
            )

        raise FileNotFoundError("\n".join(message))

    # --------------------------------------------------------
    # قراءة الصورة الأصلية
    # --------------------------------------------------------

    original_image = cv2.imread(str(IMAGE_PATH))

    if original_image is None:
        raise RuntimeError(
            "OpenCV وجد الملف، لكنه لم يتمكن من قراءة الصورة:\n"
            f"{IMAGE_PATH}"
        )

    print("\nOriginal image size:")
    print(
        f"Width={original_image.shape[1]}, "
        f"Height={original_image.shape[0]}"
    )

    # ========================================================
    # 1. Image Preprocessing
    # ========================================================

    print("\n[1/3] Running Image Preprocessing...")

    preprocess_result = preprocess_pipeline(
        image_bgr=original_image,
        wood_width_mm=WOOD_WIDTH_MM,
        wood_height_mm=WOOD_HEIGHT_MM,
        tool_dia_mm=TOOL_DIAMETER_MM,
        threshold_method="otsu",
        use_clahe=False,
        fit_mode="contain",
    )

    binary_image = preprocess_result.binary

    print("Preprocessing completed.")
    print(f"pixel_to_mm = {preprocess_result.pixel_to_mm:.6f}")
    print(f"padding     = {preprocess_result.pad_px} px")

    for note in preprocess_result.scale_notes:
        print("[Preprocessing note]", note)

    # ========================================================
    # 2. Contour Extraction
    # ========================================================

    print("\n[2/3] Running Contour Extraction...")

    extracted_contours, extraction_report = contour_extraction_pipeline(
        binary=binary_image,
        tool_dia_mm=TOOL_DIAMETER_MM,
        pixel_to_mm=preprocess_result.pixel_to_mm,
        min_area_px=None,
        classify_curves=True,
        is_tiled_pattern=False,
    )

    print("Contour extraction completed.")

    print("\n--- Contour Extraction Report ---")
    print(f"Total found          : {extraction_report.total_found}")
    print(f"Kept                 : {extraction_report.kept}")
    print(f"Dropped too small    : {extraction_report.dropped_too_small}")
    print(f"Dropped background   : {extraction_report.dropped_as_background}")
    print(f"Dropped tile seams   : {extraction_report.dropped_as_tile_seam}")

    outer_count = sum(
        1 for contour in extracted_contours
        if not contour.is_hole
    )

    hole_count = sum(
        1 for contour in extracted_contours
        if contour.is_hole
    )

    print(f"Outer contours       : {outer_count}")
    print(f"Hole contours        : {hole_count}")

    for note in extraction_report.notes:
        if note:
            print("[Contour note]", note)

    if not extracted_contours:
        raise RuntimeError(
            "لم يتم استخراج أي contour، لذلك لا يمكن تشغيل DPHull."
        )

    # ========================================================
    # 3. DPHull Simplification
    # ========================================================

    print("\n[3/3] Running DPHull Simplification...")

    # contour_extraction.Contour و contour_pipeline.Contour
    # لهما نفس الحقول، لكنهما كلاسّان مختلفان.
    # لذلك نحول الكونتورات بشكل صريح.
    simplification_input = []

    for contour in extracted_contours:
        simplification_input.append(
            SimplificationContour(
                points=list(contour.points),
                closed=contour.closed,
                is_hole=contour.is_hole,
                contour_id=contour.contour_id,
                metadata=contour.metadata,
            )
        )

    # نقاط الكونتور حاليًا بوحدة البكسل.
    # DPHull يستقبل epsilon بالميليمتر ويحوّله إلى بكسل.
    pixels_per_mm = 1.0 / preprocess_result.pixel_to_mm

    simplified_contours, simplification_reports = simplify_pipeline(
        contours=simplification_input,
        epsilon_mm=DPHULL_EPSILON_MM,
        pixels_per_mm=pixels_per_mm,
        min_segment_mm=MIN_SEGMENT_MM,
        validate=True,
    )

    print("DPHull simplification completed.")

    total_points_before = sum(
        report.input_points
        for report in simplification_reports
    )

    total_points_after = sum(
        report.output_points
        for report in simplification_reports
    )

    removed_points = total_points_before - total_points_after

    if total_points_before > 0:
        reduction_percentage = (
            removed_points / total_points_before
        ) * 100.0
    else:
        reduction_percentage = 0.0

    fallback_count = sum(
        report.fell_back_to_original
        for report in simplification_reports
    )

    print("\n--- DPHull Report ---")
    print(f"Epsilon                  : {DPHULL_EPSILON_MM} mm")
    print(f"Pixels per mm            : {pixels_per_mm:.6f}")
    print(f"Contours processed       : {len(simplified_contours)}")
    print(f"Points before            : {total_points_before}")
    print(f"Points after             : {total_points_after}")
    print(f"Removed points           : {removed_points}")
    print(f"Reduction                : {reduction_percentage:.2f}%")
    print(f"Fallback to original     : {fallback_count}")

    print("\n--- Per-contour results ---")

    for report in simplification_reports:
        print(
            f"Contour {report.contour_id}: "
            f"{report.input_points} -> {report.output_points} points"
        )

        if report.had_self_intersections_before:
            print("  Warning: self-intersection before simplification")

        if report.had_self_intersections_after:
            print("  Warning: self-intersection after simplification")

        if report.fell_back_to_original:
            print("  DPHull result rejected; original contour preserved")

        for note in report.notes:
            print(f"  Note: {note}")

    # ========================================================
    # تجهيز صور النتائج
    # ========================================================

    contours_before_image = draw_contours_on_binary(
        binary_image,
        extracted_contours,
        thickness=2,
    )

    contours_after_image = draw_contours_on_binary(
        binary_image,
        simplified_contours,
        thickness=2,
    )

    comparison_image = draw_before_after_comparison(
        binary_image,
        extracted_contours,
        simplified_contours,
    )

    # ========================================================
    # حفظ النتائج
    # ========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_files = {
        "preprocessed": (
            OUTPUT_DIR / "pattern24_01_preprocessed.jpg",
            binary_image,
        ),
        "contours_before_dphull": (
            OUTPUT_DIR / "pattern24_02_contours_before_dphull.jpg",
            contours_before_image,
        ),
        "contours_after_dphull": (
            OUTPUT_DIR / "pattern24_03_contours_after_dphull.jpg",
            contours_after_image,
        ),
        "comparison": (
            OUTPUT_DIR / "pattern24_04_dphull_comparison.jpg",
            comparison_image,
        ),
    }

    print("\n--- Saved Images ---")

    for name, (path, image) in output_files.items():
        success = cv2.imwrite(str(path), image)

        if not success:
            raise RuntimeError(
                f"فشل حفظ الصورة: {path}"
            )

        print(f"{name}: {path}")

    # ========================================================
    # عرض النتائج
    # ========================================================

    cv2.imshow(
        "1 - Original Image",
        resize_for_display(original_image),
    )

    cv2.imshow(
        "2 - Preprocessed Binary Image",
        resize_for_display(binary_image),
    )

    cv2.imshow(
        "3 - Extracted Contours Before DPHull",
        resize_for_display(contours_before_image),
    )

    cv2.imshow(
        "4 - Simplified Contours After DPHull",
        resize_for_display(contours_after_image),
    )

    cv2.imshow(
        "5 - DPHull Comparison",
        resize_for_display(comparison_image),
    )

    print("\nComparison colors:")
    print("Blue   = contour before DPHull")
    print("Red    = contour after DPHull")
    print("Yellow = points retained by DPHull")

    print("\nاضغطي أي زر وأنتِ محددة إحدى نوافذ الصور لإغلاقها.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        print(error)

        cv2.destroyAllWindows()
        raise