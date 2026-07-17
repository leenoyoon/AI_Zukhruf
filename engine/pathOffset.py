import cv2
import numpy as np
from shapely.geometry import Polygon
import os


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