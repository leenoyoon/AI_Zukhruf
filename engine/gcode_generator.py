import cv2
import numpy as np
from shapely.geometry import Polygon, JOIN_STYLE 
import os
import math

def point_line_distance(pt, start, end):
    if np.all(start == end):
        return np.linalg.norm(pt - start)
    return np.abs(np.cross(end - start, start - pt)) / np.linalg.norm(end - start)

def custom_douglas_peucker(points, epsilon):
    dmax = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = point_line_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d
            
    if dmax > epsilon:
        res1 = custom_douglas_peucker(points[:index+1], epsilon)
        res2 = custom_douglas_peucker(points[index:], epsilon)
        return np.vstack((res1[:-1], res2))
    else:
        return np.array([points[0], points[end]])


def optimize_paths(paths):
    if not paths:
        return []
    optimized = []
    current_pos = (0, 0) 
    unvisited = paths.copy()
    while unvisited:
        best_idx = 0
        best_point_idx = 0
        min_dist = float('inf')
        for i, path in enumerate(unvisited):
            for j, pt in enumerate(path):
                dist = math.dist(current_pos, pt)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
                    best_point_idx = j
        best_path = unvisited.pop(best_idx)
        if len(best_path) > 0 and best_path[0] == best_path[-1]: 
            best_path_closed = best_path[:-1] 
            best_path_rotated = best_path_closed[best_point_idx:] + best_path_closed[:best_point_idx]
            best_path_rotated.append(best_path_rotated[0])
            optimized.append(best_path_rotated)
            current_pos = best_path_rotated[-1]
        else:
            if best_point_idx == len(best_path) - 1:
                best_path.reverse()
            optimized.append(best_path)
            current_pos = best_path[-1]
    return optimized


def process_image_to_gcode(image_path, output_path, pixel_to_mm=0.5, tool_dia=3.0, depth=-3.0, step_down=1.0):
    print(f"--- Processing: {os.path.basename(image_path)} ---")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found!")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    corners_mean = np.mean([gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]])
    if corners_mean < 127: 
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else: 
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  
    contours, hierarchy = cv2.findContours(clean_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    offset_paths = []
    tool_radius = tool_dia / 2.0
    img_area = img.shape[0] * img.shape[1]
    
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 0.95 * img_area:
                continue    
            
            cnt_2d = np.squeeze(cnt)
            if cnt_2d.ndim == 1 or len(cnt_2d) < 3:
                continue
              
            epsilon = 0.004 * cv2.arcLength(cnt, True)
            approx = custom_douglas_peucker(cnt_2d, epsilon)
            path = approx * pixel_to_mm
            
            try:
                poly = Polygon(path)
                if not poly.is_valid:
                    poly = poly.buffer(0)

                parent_idx = hierarchy[0][i][3]
                
                if parent_idx == -1:
                    offset_poly = poly.buffer(tool_radius, join_style=JOIN_STYLE.mitre)
                else:
                    offset_poly = poly.buffer(-tool_radius, join_style=JOIN_STYLE.mitre)

                if not offset_poly.is_empty:
                    if offset_poly.geom_type == 'Polygon':
                        offset_paths.append(list(offset_poly.exterior.coords))
                    elif offset_poly.geom_type == 'MultiPolygon':
                        for geom in offset_poly.geoms:
                            offset_paths.append(list(geom.exterior.coords))
            except:
                continue

    optimized_paths = optimize_paths(offset_paths)
    
    gcode = [
        "G21 (Set units to mm)",
        "G90 (Absolute positioning)",
        "M3 S1000 (Spindle ON)"
    ]
    
    for path in optimized_paths:
        start_x, start_y = path[0]
        gcode.append(f"G0 X{start_x:.3f} Y{start_y:.3f} Z5.0")
        current_z = 0.0
        while current_z > depth:
            current_z -= step_down
            if current_z < depth:
                current_z = depth
            gcode.append(f"G1 Z{current_z:.3f} F200")
            for point in path[1:]:
                px, py = point
                gcode.append(f"G1 X{px:.3f} Y{py:.3f} F800")
        gcode.append("G0 Z5.0")

    gcode.append("M5 (Spindle OFF)")
    gcode.append("G0 X0 Y0 Z10 (Return to home)")
    gcode.append("M30 (End)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(gcode))
     
    print(f"Success! {len(optimized_paths)} sharp offset paths saved to: {output_path}")