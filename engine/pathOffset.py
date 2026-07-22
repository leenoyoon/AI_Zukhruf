import cv2
import numpy as np
from shapely.geometry import Polygon
import os


# ---------------------------------------------------------------------------
# نقطة الربط الرسمية بين Stage 1 (image_preprocessing.py + contour_extraction.py
# عند لين -- استدعيها عبر engine/preprocessing_stage.py::run_stage1) ومرحلة
# Path Offsetting هون.
#
# ليش دالة جديدة منفصلة، مش تعديل على process_image_to_offset_paths تحت؟
# 1) process_image_to_offset_paths بتاخد مسار صورة وبتعيد عمل threshold +
#    findContours من الصفر -- هاد بالضبط الشغل يلي صار مكرر ثلاث مرات بالمشروع
#    (هون، بـ gcode_generator.py، وبكود لين). بعد ما يصير الربط، لازم
#    process_image_to_offset_paths تنحذف أو تتحول لـ wrapper رفيع فوق
#    offset_contours -- بس هاد قرار لازم ياخد بالتنسيق معاكم كفريق لأنو
#    فيه كود تاني (pathOptimizstion.py) بيعتمد على السلوك الحالي وقت الـ import.
# 2) بيصلح ملاحظة الدكتور #9 (نفس قصة #2): process_image_to_offset_paths
#    تحت كانت عم تعمل buffer(-tool_radius) لكل الأشكال بلا استثناء، لأنها
#    ما كانت تستخدم hierarchy إطلاقاً (ولا حتى بتطلبه من findContours).
#    هون منستخدم is_hole الجاهزة من contour_extraction.py (مبنية صح على
#    hierarchy[...,3] حسب Suzuki & Abe, Property 2):
#        is_hole=False (حد خارجي)  -> buffer(+tool_radius)  (توسيع للخارج)
#        is_hole=True  (فجوة داخلية) -> buffer(-tool_radius) (تضييق للداخل)
# ---------------------------------------------------------------------------
def offset_contours(contours, pixel_to_mm: float, tool_dia_mm: float):
    """
    Parameters
    ----------
    contours : List[Contour]   -- ناتج contour_extraction_pipeline (إحداثيات
                                   بالبكسل ضمن الصورة المعالجة/المحشوة، مع
                                   is_hole جاهزة).
    pixel_to_mm : float        -- من PreprocessResult.pixel_to_mm (نفس القيمة
                                   المستخدمة بمرحلة Stage 1، وحيدة وموحّدة X/Y).
    tool_dia_mm : float        -- قطر الأداة بالملم.

    Returns
    -------
    List[List[(x, y)]] بالملم -- **نفس الشكل تماماً** يلي process_image_to_offset_paths
    كانت بترجعه (shapely exterior.coords، أول نقطة مكررة بالآخر)، فـ
    dphull_integration.py و pathOptimizstion.py و generate_Gcode.py ما
    بيحتاجوا أي تعديل ليقبلوا هالخرج.
    """
    offset_paths = []
    tool_radius = tool_dia_mm / 2.0

    for c in contours:
        pts_mm = [(x * pixel_to_mm, y * pixel_to_mm) for x, y in c.points]
        if len(pts_mm) < 3:
            continue

        try:
            poly = Polygon(pts_mm)
            if not poly.is_valid:
                poly = poly.buffer(0)

            signed_radius = -tool_radius if c.is_hole else tool_radius
            offset_poly = poly.buffer(signed_radius)

            if offset_poly.is_empty:
                continue

            if offset_poly.geom_type == "Polygon":
                offset_paths.append(list(offset_poly.exterior.coords))
            elif offset_poly.geom_type == "MultiPolygon":
                for geom in offset_poly.geoms:
                    offset_paths.append(list(geom.exterior.coords))
        except Exception:
            continue

    print(f"[offset_contours] {len(offset_paths)} offset path(s) built from "
          f"{len(contours)} input contour(s) from Stage 1")
    return offset_paths


def process_image_to_offset_paths(
    image_path,
    pixel_to_mm=0.5,
    tool_dia=3.0
):
    print(f"--- Processing: {os.path.basename(image_path)} ---")

    offset_paths = []

    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found!")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    corners_mean = np.mean([
        gray[0, 0],
        gray[0, -1],
        gray[-1, 0],
        gray[-1, -1]
    ])

    if corners_mean < 127:
        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    kernel = np.ones((3, 3), np.uint8)

    clean_binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        clean_binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    raw_paths = []

    img_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        if cv2.contourArea(cnt) > 0.95 * img_area:
            continue

        epsilon = 0.001 * cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            epsilon,
            True
        )

        points_mm = np.squeeze(approx) * pixel_to_mm

        if len(points_mm.shape) == 1 or len(points_mm) < 3:
            continue

        raw_paths.append(points_mm)

    tool_radius = tool_dia / 2.0

    for path in raw_paths:
        try:
            poly = Polygon(path)

            if not poly.is_valid:
                poly = poly.buffer(0)

            offset_poly = poly.buffer(-tool_radius)

            if offset_poly.is_empty:
                continue

            if offset_poly.geom_type == "Polygon":
                offset_paths.append(
                    list(offset_poly.exterior.coords)
                )

            elif offset_poly.geom_type == "MultiPolygon":
                for geom in offset_poly.geoms:
                    offset_paths.append(
                        list(geom.exterior.coords)
                    )

        except Exception:
            continue

    print(f"Extracted {len(offset_paths)} offset paths")

    return offset_paths

offset_paths = process_image_to_offset_paths(
    image_path="data/input_images/pattern24.png",
    pixel_to_mm=0.5,
    tool_dia=3.0
)