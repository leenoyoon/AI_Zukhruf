import matplotlib.pyplot as plt

from pathOffset import process_image_to_offset_paths


def plot_offset_paths(offset_paths, title="Offset Paths"):
    plt.figure(figsize=(10, 10))

    for path in offset_paths:
        if len(path) < 2:
            continue

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        plt.plot(xs, ys, linewidth=1)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis()  # لأن إحداثيات الصورة تبدأ من الأعلى
    plt.title(title)
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True)
    plt.show()


offset_paths = process_image_to_offset_paths(
    image_path="data/input_images/pattern24.png",
    pixel_to_mm=0.5,
    tool_dia=3.0
)

plot_offset_paths(offset_paths)