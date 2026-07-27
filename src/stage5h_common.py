from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import torch


TARGET_TO_ID = {
    "right_elbow": 0,
    "left_elbow": 1,
    "right_knee": 2,
    "left_knee": 3,
}

ID_TO_TARGET = {v: k for k, v in TARGET_TO_ID.items()}

ANCHOR_NAMES = [
    "right_shoulder",
    "right_wrist",
    "right_hip",
    "right_ankle",
    "left_shoulder",
    "left_wrist",
    "left_hip",
    "left_ankle",
]

TARGET_TO_ANCHOR_NAMES = {
    "right_elbow": ("right_shoulder", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_wrist"),
    "right_knee": ("right_hip", "right_ankle"),
    "left_knee": ("left_hip", "left_ankle"),
}

TARGET_TO_ANCHOR_IDXS = {
    "right_elbow": (0, 1),
    "left_elbow": (4, 5),
    "right_knee": (2, 3),
    "left_knee": (6, 7),
}

MPII_ID = {
    "right_ankle": 0,
    "right_knee": 1,
    "right_hip": 2,
    "left_hip": 3,
    "left_knee": 4,
    "left_ankle": 5,
    "right_wrist": 10,
    "right_elbow": 11,
    "right_shoulder": 12,
    "left_shoulder": 13,
    "left_elbow": 14,
    "left_wrist": 15,
}

TARGET_TO_MISSING_MPII = {
    "right_elbow": 11,
    "left_elbow": 14,
    "right_knee": 1,
    "left_knee": 4,
}


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def dist2d(x1, y1, x2, y2):
    vals = [x1, y1, x2, y2]
    if any(pd.isna(v) for v in vals):
        return np.nan
    return float(np.sqrt((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2))


def make_heatmap(x_norm, y_norm, h, w, sigma=2.0):
    heat = np.zeros((h, w), dtype=np.float32)

    if not np.isfinite(x_norm) or not np.isfinite(y_norm):
        return heat

    if x_norm < 0 or y_norm < 0 or x_norm > 1 or y_norm > 1:
        return heat

    cx = x_norm * (w - 1)
    cy = y_norm * (h - 1)
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    return heat.astype(np.float32)


def soft_argmax_np(heat):
    h, w = heat.shape
    flat = heat.reshape(-1)
    if flat.max() <= 0:
        return 0.5, 0.5, 0.0
    prob = np.exp(flat - flat.max())
    prob = prob / (prob.sum() + 1e-9)
    ys, xs = np.mgrid[0:h, 0:w]
    x = float((prob * xs.reshape(-1)).sum() / max(w - 1, 1))
    y = float((prob * ys.reshape(-1)).sum() / max(h - 1, 1))
    conf = float(flat.max())
    return x, y, conf


def crop_and_resize(image, x1, y1, x2, y2, input_size):
    crop = image[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        raise ValueError("Empty crop")
    crop_resized = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return crop_resized


def image_to_tensor(rgb):
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0


def get_gt_anchor_pair(row):
    target = row["target_name"]
    a_name, b_name = TARGET_TO_ANCHOR_NAMES[target]
    a_id = MPII_ID[a_name]
    b_id = MPII_ID[b_name]

    ax = safe_float(row.get(f"j{a_id}_orig_x", np.nan))
    ay = safe_float(row.get(f"j{a_id}_orig_y", np.nan))
    bx = safe_float(row.get(f"j{b_id}_orig_x", np.nan))
    by = safe_float(row.get(f"j{b_id}_orig_y", np.nan))

    return a_name, b_name, ax, ay, bx, by


def get_gt_target(row):
    return safe_float(row["target_orig_x"]), safe_float(row["target_orig_y"])


def get_teacher_anchor_full(row, name):
    x = safe_float(row.get(f"teacher_{name}_x", np.nan))
    y = safe_float(row.get(f"teacher_{name}_y", np.nan))
    c = safe_float(row.get(f"teacher_{name}_conf", 1.0), 1.0)
    return x, y, c


def build_teacher_anchor_array(row, img_w, img_h):
    xy = np.zeros((len(ANCHOR_NAMES), 2), dtype=np.float32)
    conf = np.zeros((len(ANCHOR_NAMES),), dtype=np.float32)

    for i, name in enumerate(ANCHOR_NAMES):
        x, y, c = get_teacher_anchor_full(row, name)
        if np.isfinite(x) and np.isfinite(y):
            xy[i, 0] = np.clip(x / max(img_w, 1), 0.0, 1.0)
            xy[i, 1] = np.clip(y / max(img_h, 1), 0.0, 1.0)
            conf[i] = np.clip(c, 0.0, 1.0)
        else:
            xy[i, :] = 0.5
            conf[i] = 0.0

    return xy, conf


def merge_crop_and_strong(crop_csv, strong_csv):
    crop = pd.read_csv(crop_csv)
    strong = pd.read_csv(strong_csv)

    # Avoid duplicating many prediction columns; keep GT MPII joints and id columns.
    gt_cols = ["occluded_image_name", "target_name"]
    for j in range(16):
        for suffix in ["orig_x", "orig_y", "mask"]:
            c = f"j{j}_{suffix}"
            if c in strong.columns:
                gt_cols.append(c)

    gt_cols = [c for c in gt_cols if c in strong.columns]
    strong_gt = strong[gt_cols].copy()

    out = crop.merge(strong_gt, on=["occluded_image_name", "target_name"], how="left")
    return out


def split_by_image(df, val_ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)

    if "image_name" in df.columns:
        names = df["image_name"].fillna(df["occluded_image_name"]).unique()
        key = df["image_name"].fillna(df["occluded_image_name"])
    else:
        names = df["occluded_image_name"].unique()
        key = df["occluded_image_name"]

    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])

    val_df = df[key.isin(val_names)].copy()
    train_df = df[~key.isin(val_names)].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
