import os

from quickdraw import QuickDrawDataGroup

GROUPS = {
    "animals": ["ant", "bear", "bee", "bird", "butterfly", "camel", "cat", "cow", "dog", "dolphin", "elephant", "fish", "frog", "giraffe", "lion", "monkey", "mouse", "owl", "panda", "penguin", "pig", "rabbit", "sheep", "snake", "spider", "tiger", "whale", "zebra"],
    "fruits": ["apple", "banana", "blackberry", "blueberry", "grapes", "pear", "pineapple", "strawberry", "watermelon"],
    "vehicles": ["airplane", "bicycle", "bus", "car", "helicopter", "pickup truck", "police car", "school bus", "speedboat", "train", "truck", "van"],
    "flowers": ["flower", "cactus", "house plant"]
}

ALL_CLASSES = [cls for group in GROUPS.values() for cls in group]
class_to_id = {cls: i for i, cls in enumerate(ALL_CLASSES)}

SAMPLES_PER_TOPIC = 1000
IMG_SIZE = 256

dataset_dir = "quickdraw_data"
dirpath = f"../data/{dataset_dir}"


def get_yolo_format(drawing, img_size=256):
    """
    Tính toán tọa độ YOLO (x_center, y_center, width, height) từ strokes
    QuickDraw mặc định dùng canvas 256x256
    """
    all_x = []
    all_y = []

    for stroke in drawing.strokes:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])

    if not all_x or not all_y:
        return None

    # 1. Tìm cực trị (pixel)
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    # 2. Tính toán center và size (pixel)
    width_px = xmax - xmin
    height_px = ymax - ymin
    x_center_px = xmin + (width_px / 2)
    y_center_px = ymin + (height_px / 2)

    # 3. Chuẩn hóa về dải [0, 1] cho YOLO
    # (Dữ liệu QuickDraw mặc định nằm trong khung 256x256)
    x_center = x_center_px / img_size
    y_center = y_center_px / img_size
    w = width_px / img_size
    h = height_px / img_size

    return x_center, y_center, w, h


def process_drawing(drawing_data, file_name, cid, label):
    os.makedirs(f"{dirpath}/{label}/images", exist_ok=True)
    img_path = f"{dirpath}/{label}/images/{file_name}.jpg"
    yolo_bounds = get_yolo_format(drawing_data, img_size=IMG_SIZE)
    if yolo_bounds is None:
        return
    xc, yc, w, h = yolo_bounds
    img = drawing_data.get_image(stroke_width=3).resize((IMG_SIZE, IMG_SIZE))
    img.save(img_path)
    os.makedirs(f"{dirpath}/{label}/labels", exist_ok=True)
    label_path = f"{dirpath}/{label}/labels/{file_name}.txt"
    with open(label_path, "w") as f:
        f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

for label in ALL_CLASSES:
    print(f"Đang tải {label}...")
    # Thư viện này sẽ tự động tìm link tải hoạt động
    group = QuickDrawDataGroup(label, max_drawings=SAMPLES_PER_TOPIC, recognized=True)

    for i, drawing in enumerate(group.drawings):
        name = f"{label}_{i}"
        cid = class_to_id[label]
        process_drawing(drawing, name, cid, label)
print("Xong!")