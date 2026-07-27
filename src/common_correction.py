import json
from pathlib import Path

import numpy as np


TARGET_TO_COCO = {
    "right_elbow": 8,
    "left_elbow": 7,
    "right_knee": 14,
    "left_knee": 13,
}

TARGET_TO_TASK_ID = {
    "right_elbow": 0,
    "left_elbow": 1,
    "right_knee": 2,
    "left_knee": 3,
}

ANCHOR_COCO = {
    "right_elbow": (6, 10),  # right shoulder, right wrist
    "left_elbow": (5, 9),    # left shoulder, left wrist
    "right_knee": (12, 16),  # right hip, right ankle
    "left_knee": (11, 15),   # left hip, left ankle
}


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def one_hot(index, size):
    arr = np.zeros(size, dtype=np.float32)
    arr[index] = 1.0
    return arr


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def get_anchor_conf(row, target_name):
    a, b = ANCHOR_COCO[target_name]
    ca = safe_float(row.get(f"coco{a}_conf", 0.0), 0.0)
    cb = safe_float(row.get(f"coco{b}_conf", 0.0), 0.0)
    return min(ca, cb), ca, cb


def compute_image_scale(row):
    """
    Estimate person/image scale from available GT skeleton coordinates.
    Used for thresholding shift.
    """
    xs = []
    ys = []

    for j in range(16):
        x = safe_float(row.get(f"j{j}_orig_x", 0.0), 0.0)
        y = safe_float(row.get(f"j{j}_orig_y", 0.0), 0.0)
        if x > 0 and y > 0:
            xs.append(x)
            ys.append(y)

    if len(xs) < 3:
        return 1.0

    return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)


def point_error(px, py, tx, ty):
    if np.isnan(px) or np.isnan(py):
        return np.nan
    return float(np.sqrt((px - tx) ** 2 + (py - ty) ** 2))


def normalize_point(x, y, width, height):
    return float(x) / max(float(width), 1.0), float(y) / max(float(height), 1.0)


def denormalize_point(nx, ny, width, height):
    return float(nx) * float(width), float(ny) * float(height)


def infer_width_height_from_row(row):
    """
    We may not have actual image width/height columns.
    Estimate canvas size from known coordinates.
    This is enough for coordinate normalization.
    """
    xs = []
    ys = []

    keys = ["target_orig_x", "yolo_target_x", "prior_pred_x", "geometry_x"]
    for k in keys:
        v = safe_float(row.get(k, 0.0), 0.0)
        if v > 0:
            xs.append(v)

    keys_y = ["target_orig_y", "yolo_target_y", "prior_pred_y", "geometry_y"]
    for k in keys_y:
        v = safe_float(row.get(k, 0.0), 0.0)
        if v > 0:
            ys.append(v)

    for k in range(17):
        x = safe_float(row.get(f"coco{k}_x", 0.0), 0.0)
        y = safe_float(row.get(f"coco{k}_y", 0.0), 0.0)
        if x > 0 and y > 0:
            xs.append(x)
            ys.append(y)

    for j in range(16):
        x = safe_float(row.get(f"j{j}_orig_x", 0.0), 0.0)
        y = safe_float(row.get(f"j{j}_orig_y", 0.0), 0.0)
        if x > 0 and y > 0:
            xs.append(x)
            ys.append(y)

    width = max(xs) + 50 if xs else 640
    height = max(ys) + 50 if ys else 480
    return float(width), float(height)


def build_feature(row):
    """
    Build correction-head feature vector.

    All coordinates are normalized by estimated image width/height.
    """

    width, height = infer_width_height_from_row(row)
    target_name = row["target_name"]
    target_task = TARGET_TO_TASK_ID[target_name]

    feat = []

    # 17 COCO keypoints: x,y,conf
    for k in range(17):
        x = safe_float(row.get(f"coco{k}_x", 0.0), 0.0)
        y = safe_float(row.get(f"coco{k}_y", 0.0), 0.0)
        c = safe_float(row.get(f"coco{k}_conf", 0.0), 0.0)

        nx, ny = normalize_point(x, y, width, height)
        feat.extend([nx, ny, c])

    # Main candidate points
    candidates = [
        ("yolo_target_x", "yolo_target_y"),
        ("prior_pred_x", "prior_pred_y"),
        ("geometry_x", "geometry_y"),
    ]

    for xk, yk in candidates:
        x = safe_float(row.get(xk, 0.0), 0.0)
        y = safe_float(row.get(yk, 0.0), 0.0)
        nx, ny = normalize_point(x, y, width, height)
        exists = 1.0 if x > 0 and y > 0 else 0.0
        feat.extend([nx, ny, exists])

    # Confidence and trust features
    target_coco = TARGET_TO_COCO[target_name]
    target_conf = safe_float(row.get(f"coco{target_coco}_conf", row.get("yolo_target_conf", 0.0)), 0.0)
    anchor_min, anchor_a, anchor_b = get_anchor_conf(row, target_name)

    yolo_to_prior = 0.0
    yx = safe_float(row.get("yolo_target_x", 0.0), 0.0)
    yy = safe_float(row.get("yolo_target_y", 0.0), 0.0)
    px = safe_float(row.get("prior_pred_x", 0.0), 0.0)
    py = safe_float(row.get("prior_pred_y", 0.0), 0.0)
    if yx > 0 and yy > 0 and px > 0 and py > 0:
        yolo_to_prior = np.sqrt((yx - px) ** 2 + (yy - py) ** 2) / max(compute_image_scale(row), 1.0)

    feat.extend([
        target_conf,
        anchor_min,
        anchor_a,
        anchor_b,
        yolo_to_prior,
    ])

    # Target one-hot
    feat.extend(one_hot(target_task, 4))

    target_x = safe_float(row.get("target_orig_x", 0.0), 0.0)
    target_y = safe_float(row.get("target_orig_y", 0.0), 0.0)
    target = np.array(normalize_point(target_x, target_y, width, height), dtype=np.float32)

    meta = {
        "width": width,
        "height": height,
    }

    return np.array(feat, dtype=np.float32), target, meta
