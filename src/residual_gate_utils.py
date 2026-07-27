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


def point_error(px, py, tx, ty):
    if np.isnan(px) or np.isnan(py):
        return np.nan
    return float(np.sqrt((px - tx) ** 2 + (py - ty) ** 2))


def estimate_canvas(row):
    xs = []
    ys = []

    keys_x = ["target_orig_x", "yolo_target_x", "prior_pred_x", "geometry_x"]
    keys_y = ["target_orig_y", "yolo_target_y", "prior_pred_y", "geometry_y"]

    for k in keys_x:
        v = safe_float(row.get(k, 0.0), 0.0)
        if v > 0:
            xs.append(v)

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

    width = max(xs) + 50 if xs else 640.0
    height = max(ys) + 50 if ys else 480.0

    return float(width), float(height)


def person_scale(row):
    xs = []
    ys = []

    for j in range(16):
        x = safe_float(row.get(f"j{j}_orig_x", 0.0), 0.0)
        y = safe_float(row.get(f"j{j}_orig_y", 0.0), 0.0)
        if x > 0 and y > 0:
            xs.append(x)
            ys.append(y)

    if len(xs) < 3:
        return 100.0

    return float(max(max(xs) - min(xs), max(ys) - min(ys), 1.0))


def normalize_xy(x, y, width, height):
    return float(x) / max(width, 1.0), float(y) / max(height, 1.0)


def denormalize_xy(nx, ny, width, height):
    return float(nx) * width, float(ny) * height


def get_anchor_conf(row, target_name):
    a, b = ANCHOR_COCO[target_name]

    ca = safe_float(row.get(f"coco{a}_conf", 0.0), 0.0)
    cb = safe_float(row.get(f"coco{b}_conf", 0.0), 0.0)

    return min(ca, cb), ca, cb


def inside_box(x, y, x1, y1, x2, y2):
    if x <= 0 or y <= 0:
        return 0.0
    return 1.0 if (x1 <= x <= x2 and y1 <= y <= y2) else 0.0


def build_feature_and_targets(row, bad_error_px=50.0):
    """
    Feature vector for residual-gate model.

    Output targets:
    - residual target in normalized coordinates:
      true_norm - yolo_norm
    - gate label:
      1 if YOLO error is larger than bad_error_px else 0
    """

    width, height = estimate_canvas(row)
    scale = person_scale(row)

    target_name = row["target_name"]
    target_task_id = TARGET_TO_TASK_ID[target_name]
    target_coco_id = TARGET_TO_COCO[target_name]

    feat = []

    # 17 COCO keypoints: normalized x,y, confidence
    for k in range(17):
        x = safe_float(row.get(f"coco{k}_x", 0.0), 0.0)
        y = safe_float(row.get(f"coco{k}_y", 0.0), 0.0)
        c = safe_float(row.get(f"coco{k}_conf", 0.0), 0.0)
        nx, ny = normalize_xy(x, y, width, height)
        feat.extend([nx, ny, c])

    # main points: YOLO, prior, geometry
    point_names = [
        ("yolo_target_x", "yolo_target_y"),
        ("prior_pred_x", "prior_pred_y"),
        ("geometry_x", "geometry_y"),
    ]

    for xk, yk in point_names:
        x = safe_float(row.get(xk, 0.0), 0.0)
        y = safe_float(row.get(yk, 0.0), 0.0)
        nx, ny = normalize_xy(x, y, width, height)
        exists = 1.0 if x > 0 and y > 0 else 0.0
        feat.extend([nx, ny, exists])

    # distances between candidates, normalized by person scale
    yx = safe_float(row.get("yolo_target_x", 0.0), 0.0)
    yy = safe_float(row.get("yolo_target_y", 0.0), 0.0)

    px = safe_float(row.get("prior_pred_x", 0.0), 0.0)
    py = safe_float(row.get("prior_pred_y", 0.0), 0.0)

    gx = safe_float(row.get("geometry_x", 0.0), 0.0)
    gy = safe_float(row.get("geometry_y", 0.0), 0.0)

    def dist(a, b, c, d):
        if a <= 0 or b <= 0 or c <= 0 or d <= 0:
            return 0.0
        return float(np.sqrt((a - c) ** 2 + (b - d) ** 2) / max(scale, 1.0))

    feat.extend([
        dist(yx, yy, px, py),
        dist(yx, yy, gx, gy),
        dist(px, py, gx, gy),
    ])

    # confidence features
    target_conf = safe_float(row.get("yolo_target_conf", row.get(f"coco{target_coco_id}_conf", 0.0)), 0.0)
    anchor_min, anchor_a, anchor_b = get_anchor_conf(row, target_name)

    feat.extend([
        target_conf,
        anchor_min,
        anchor_a,
        anchor_b,
    ])

    # occlusion box features
    x1 = safe_float(row.get("occ_x1", 0.0), 0.0)
    y1 = safe_float(row.get("occ_y1", 0.0), 0.0)
    x2 = safe_float(row.get("occ_x2", 0.0), 0.0)
    y2 = safe_float(row.get("occ_y2", 0.0), 0.0)

    occ_cx = (x1 + x2) / 2.0
    occ_cy = (y1 + y2) / 2.0
    occ_w = max(x2 - x1, 1.0)
    occ_h = max(y2 - y1, 1.0)

    occ_cx_n, occ_cy_n = normalize_xy(occ_cx, occ_cy, width, height)
    feat.extend([
        occ_cx_n,
        occ_cy_n,
        occ_w / max(width, 1.0),
        occ_h / max(height, 1.0),
        inside_box(yx, yy, x1, y1, x2, y2),
        inside_box(px, py, x1, y1, x2, y2),
        inside_box(gx, gy, x1, y1, x2, y2),
    ])

    # target one-hot
    feat.extend(one_hot(target_task_id, 4))

    # targets
    tx = safe_float(row.get("target_orig_x", 0.0), 0.0)
    ty = safe_float(row.get("target_orig_y", 0.0), 0.0)

    yx_n, yy_n = normalize_xy(yx, yy, width, height)
    tx_n, ty_n = normalize_xy(tx, ty, width, height)

    residual_target = np.array([tx_n - yx_n, ty_n - yy_n], dtype=np.float32)

    yolo_error = safe_float(row.get("yolo_target_error_px", point_error(yx, yy, tx, ty)), 0.0)
    gate_target = 1.0 if yolo_error >= bad_error_px else 0.0

    meta = {
        "width": width,
        "height": height,
        "scale": scale,
        "yolo_norm_x": yx_n,
        "yolo_norm_y": yy_n,
    }

    return np.array(feat, dtype=np.float32), residual_target, np.float32(gate_target), meta
