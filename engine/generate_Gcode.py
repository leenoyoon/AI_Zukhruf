

def generate_gcode(
    optimized_paths,
    safe_z=5.0,
    cut_depth=-2.0,
    feed_rate=800,
    spindle_speed=12000,
    arc_error=0.15
):
    gcode = []

    gcode.append("G21 ; units in mm")
    gcode.append("G90 ; absolute positioning")
    gcode.append(f"M3 S{spindle_speed} ; spindle on")
    gcode.append(f"G0 Z{safe_z:.3f}")

    for path in optimized_paths:

        points = path["points"]

        if len(points) < 2:
            continue

        start = points[0]

        # Move above start point
        gcode.append(
            f"G0 X{start[0]:.3f} Y{start[1]:.3f}"
        )

        # Plunge
        gcode.append(
            f"G1 Z{cut_depth:.3f} F{feed_rate}"
        )

        # Build quick lookup for arc ranges
        arc_segments = path.get("arc_segments", [])

        arc_ranges = []

        for arc in arc_segments:
            arc_points = arc["points"]

            start_idx = None
            end_idx = None

            # find indices inside original points
            for i in range(len(points)):
                if points[i] == arc_points[0]:
                    start_idx = i
                    break

            for i in range(len(points)):
                if points[i] == arc_points[-1]:
                    end_idx = i
                    break

            if start_idx is not None and end_idx is not None:
                arc_ranges.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "arc": arc
                })

        i = 0

        while i < len(points) - 1:

            used_arc = False

            # check if current index starts an arc
            for arc_data in arc_ranges:

                if i == arc_data["start_idx"]:

                    arc = arc_data["arc"]

                    end = arc["end"]
                    center = arc["center"]
                    start_point = arc["start"]

                    i_offset = center[0] - start_point[0]
                    j_offset = center[1] - start_point[1]

                    gcode.append(
                        f"{arc['command']} "
                        f"X{end[0]:.3f} "
                        f"Y{end[1]:.3f} "
                        f"I{i_offset:.3f} "
                        f"J{j_offset:.3f} "
                        f"F{feed_rate}"
                    )

                    i = arc_data["end_idx"]

                    used_arc = True
                    break

            if used_arc:
                continue

            # otherwise generate normal line
            end = points[i + 1]

            gcode.append(
                f"G1 X{end[0]:.3f} "
                f"Y{end[1]:.3f} "
                f"F{feed_rate}"
            )

            i += 1

        # Lift tool
        gcode.append(f"G0 Z{safe_z:.3f}")

    gcode.append("M5 ; spindle off")
    gcode.append("G0 X0 Y0 ; return home")
    gcode.append("M30 ; end program")

    return "\n".join(gcode)




# print(gcode_output)


# output_path = "output_gcode/optimized_toolpath.nc"

# with open(output_path, "w") as file:
#     file.write(gcode_output)

# print(f"G-code saved to: {output_path}")