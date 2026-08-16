
import os
import math
import logging
import plotly.graph_objects as go
from config import Config  # استيراد إعدادات المسارات

logging.basicConfig(level=logging.INFO, format="[GCode-Sim] %(levelname)s: %(message)s")

# المسار الافتراضي للحفظ داخل مجلد output_simulations
DEFAULT_SIM_PATH = os.path.join(Config.SIMULATION_DIR, "simulation.html")

# عدد النقاط المستخدمة لتقريب كل قوس (كلما زاد الرقم، صار القوس أنعم بصريًا)
ARC_RESOLUTION = 24

# أوامر الحركة المعترف بها، مع تحديد إذا كانت قوس (True) أو خط/رابيد (False)
MOTION_COMMANDS = {
    "G0": False, "G00": False,
    "G1": False, "G01": False,
    "G2": True, "G02": True,
    "G3": True, "G03": True,
}


def _interpolate_arc(start_x, start_y, i_off, j_off, end_x, end_y, clockwise):
    """
    Computes intermediate (x, y) points along a G2/G3 arc so it renders as a
    real curve instead of a straight chord between its start and end points.

    NOTE ON I/J CONVENTION: this assumes I/J are RELATIVE offsets from the arc's
    start point (center = start + (I, J)) — the standard default on the vast
    majority of CNC controllers (G91.1-style incremental IJ), and the convention
    always emitted by our own gcode_generator. If you ever feed this simulator
    G-code from a source that uses ABSOLUTE I/J (G90.1), the arcs will render
    incorrectly and need a separate code path.

    Returns a list of points EXCLUDING the start point (already in the path)
    and INCLUDING the end point.
    """
    center_x = start_x + i_off
    center_y = start_y + j_off
    radius = math.hypot(i_off, j_off)

    if radius < 1e-6:
        # لا يوجد إزاحة مركز صالحة -> ما فينا نحسب قوس، نرجع لخط مستقيم كحل آمن
        return [(end_x, end_y)]

    start_angle = math.atan2(start_y - center_y, start_x - center_x)
    end_angle = math.atan2(end_y - center_y, end_x - center_x)

    if clockwise:  # G2
        if end_angle >= start_angle:
            end_angle -= 2 * math.pi
    else:  # G3
        if end_angle <= start_angle:
            end_angle += 2 * math.pi

    points = []
    for step in range(1, ARC_RESOLUTION + 1):
        t = step / ARC_RESOLUTION
        angle = start_angle + (end_angle - start_angle) * t
        px = center_x + radius * math.cos(angle)
        py = center_y + radius * math.sin(angle)
        points.append((px, py))

    return points


def generate_gcode_simulation_html(
    gcode_text,
    output_html_path=DEFAULT_SIM_PATH,
    offline_ready=False,
    wood_width_mm=None,      # << جديد
    wood_height_mm=None,     # << جديد
):
    """
    Parses G-code text and creates a fast, interactive 3D simulation plot.

    Features:
    - Correct handling of MODAL motion commands: a line with only coordinate
      words (e.g. "X20 Y10") inherits the last explicit G0/G1/G2/G3 command,
      exactly as required by the G-code standard, instead of being skipped.
    - Accurate G2/G3 arc rendering via interpolation (see _interpolate_arc).
    - Sequential 4-color gradient (Yellow -> Orange -> Red -> Purple).
    - Clean Hover Tooltips and colored Start/End text markers.

    Args:
        gcode_text: the raw G-code string to simulate.
        output_html_path: where to save the resulting HTML file.
        offline_ready: if True, embeds the full Plotly.js library in the HTML
            (~3MB, works with zero internet connection at the machine/shop
            floor). If False (default), loads Plotly.js from a CDN
            (~300KB file, but requires internet access to view the preview).
    """
    x_coords, y_coords, z_coords = [0.0], [0.0], [0.0]
    hover_texts = ["<b>Point:</b> #0<br><b>Cmd:</b> START<br><b>X:</b> 0.00 | <b>Y:</b> 0.00 | <b>Z:</b> 0.00"]
    curr_x, curr_y, curr_z = 0.0, 0.0, 0.0
    step_counter = 0
    skipped_lines = 0

    # يحفظ آخر أمر حركة صريح (G0/G1/G2/G3) لتطبيقه على الأسطر التي لا تكرره (Modal)
    active_motion_cmd = None

    lines = gcode_text.split("\n")
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.split(";")[0].strip()  # Ignore comments
        if not line:
            continue

        parts = line.split()
        first_token = parts[0].upper()

        if first_token in MOTION_COMMANDS:
            # سطر فيه أمر حركة صريح -> يصبح هو الوضع الفعّال، وباقي الرموز هي الإحداثيات
            cmd = first_token
            coord_tokens = parts[1:]
            active_motion_cmd = cmd
        elif active_motion_cmd is not None and first_token[0].upper() in "XYZIJ":
            # سطر بدون أمر G صريح لكنه إحداثيات -> يرث آخر أمر حركة فعّال (Modal Command)
            cmd = active_motion_cmd
            coord_tokens = parts
        else:
            # سطر غير متعلق بالحركة (G21, G90, M3, G4 P2 ...) -> يُتجاهل بدون كسر الحالة
            continue

        try:
            target_x, target_y, target_z = curr_x, curr_y, curr_z
            i_off, j_off = 0.0, 0.0
            has_i, has_j = False, False

            for token in coord_tokens:
                char = token[0].upper()
                value = float(token[1:])
                if char == "X":
                    target_x = value
                elif char == "Y":
                    target_y = value
                elif char == "Z":
                    target_z = value
                elif char == "I":
                    i_off = value
                    has_i = True
                elif char == "J":
                    j_off = value
                    has_j = True

            if not coord_tokens:
                continue  # سطر أمر فقط بدون إحداثيات فعلية (مثلاً "G1" لوحده)

            step_counter += 1
            is_arc = MOTION_COMMANDS[cmd]

            if is_arc and (has_i or has_j):
                clockwise = cmd in ["G2", "G02"]
                arc_pts = _interpolate_arc(
                    curr_x, curr_y, i_off, j_off, target_x, target_y, clockwise
                )
                for n, (px, py) in enumerate(arc_pts, start=1):
                    x_coords.append(px)
                    y_coords.append(py)
                    z_coords.append(target_z)
                    is_last = (n == len(arc_pts))
                    hover_texts.append(
                        f"<b>Step:</b> #{step_counter}<br>"
                        f"<b>Command:</b> {'CW Arc (G2)' if clockwise else 'CCW Arc (G3)'}"
                        f"{'' if is_last else ' (interpolated)'}<br>"
                        f"<b>X:</b> {px:.2f} mm | <b>Y:</b> {py:.2f} mm | <b>Z:</b> {target_z:.2f} mm"
                    )
                curr_x, curr_y = arc_pts[-1]
                curr_z = target_z
            else:
                curr_x, curr_y, curr_z = target_x, target_y, target_z
                x_coords.append(curr_x)
                y_coords.append(curr_y)
                z_coords.append(curr_z)
                cmd_type = "Rapid Move (G0)" if cmd in ["G0", "G00"] else f"Cut Feed ({cmd})"
                hover_texts.append(
                    f"<b>Step:</b> #{step_counter}<br>"
                    f"<b>Command:</b> {cmd_type}<br>"
                    f"<b>X:</b> {curr_x:.2f} mm | <b>Y:</b> {curr_y:.2f} mm | <b>Z:</b> {curr_z:.2f} mm"
                )

        except (ValueError, IndexError) as e:
            skipped_lines += 1
            logging.warning(f"Line {line_no} could not be parsed and was skipped ('{raw_line.strip()}'): {e}")
            continue

    if skipped_lines:
        logging.warning(f"Simulation finished with {skipped_lines} skipped/malformed line(s).")

    total_points = len(x_coords)
    step_indices = list(range(total_points))

    # التدرج الرباعي: أصفر -> برتقالي -> أحمر -> بنفسجي
    custom_colorscale = [
        [0.0, "#facc15"],   # Yellow (Start 0%)
        [0.33, "#f97316"],  # Orange (33%)
        [0.66, "#dc2626"],  # Red (66%)
        [1.0, "#9333ea"],   # Purple (End 100%)
    ]

    # 1. Trace المسار الأساسي
    path_trace = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode="lines",
        name="Toolpath",
        text=hover_texts,
        hoverinfo="text",
        line=dict(
            color=step_indices,
            colorscale=custom_colorscale,
            width=4,
            colorbar=dict(
                title=dict(text="Sequence Progression", side="right"),
                tickmode="array",
                tickvals=[
                    0,
                    int(total_points * 0.33),
                    int(total_points * 0.66),
                    max(0, total_points - 1),
                ],
                ticktext=["Start (0%)", "30%", "60%", "End (100%)"],
                len=0.35,
                x=0.01,
                y=0.98,
                xanchor="left",
                yanchor="top",
                thickness=12,
            ),
        ),
    )

    data = [path_trace]

    # 2. إضافة نقطة البداية والنهاية مع تلوين النصوص بلون النقطة
    if total_points > 0:
        start_end_trace = go.Scatter3d(
            x=[x_coords[0], x_coords[-1]],
            y=[y_coords[0], y_coords[-1]],
            z=[z_coords[0], z_coords[-1]],
            mode="markers+text",
            name="Start/End",
            marker=dict(
                size=[8, 8],
                color=["#22c55e", "#ef4444"],  # أخضر للبداية، أحمر للنهاية
                symbol="circle",
            ),
            text=["START", "END"],
            textposition="top center",
            textfont=dict(
                color=["#22c55e", "#ef4444"],
                size=12,
            ),
            hoverinfo="skip",
        )
        data.append(start_end_trace)

    fig = go.Figure(data=data)
    # ===== رسم إطار قطعة الخشب =====
    if wood_width_mm is not None and wood_height_mm is not None:
        # مستطيل على مستوى Z=0 (سطح الخشب)
        stock_x = [0, wood_width_mm, wood_width_mm, 0, 0]
        stock_y = [0, 0, wood_height_mm, wood_height_mm, 0]
        stock_z = [0, 0, 0, 0, 0]

        stock_trace = go.Scatter3d(
            x=stock_x,
            y=stock_y,
            z=stock_z,
            mode="lines",
            name="Stock",
            line=dict(color="rgba(120,120,120,0.7)", width=6, dash="dash"),
            hoverinfo="skip",
        )
        data.append(stock_trace)

        # نعيد بناء الـ Figure بعد إضافة الإطار
        fig = go.Figure(data=data)
    # ================================
    fig.update_layout(
        showlegend=False,
        title="CNC Toolpath 3D Simulation",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            yaxis=dict(autorange="reversed"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # 3. إنشاء المجلد تلقائياً في حال عدم وجوده قبل الحفظ
    output_dir = os.path.dirname(output_html_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # include_plotlyjs=True يضمّن المكتبة كاملة (~3MB) للعمل بدون إنترنت
    # "cdn" يحمّلها من الإنترنت لكن يبقي حجم الملف صغير (~300KB)
    plotly_mode = True if offline_ready else "cdn"
    fig.write_html(output_html_path, include_plotlyjs=plotly_mode)
    logging.info(f"Interactive 3D preview saved to: {output_html_path}")
    return output_html_path
