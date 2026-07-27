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

TARGET_TO_ANCHORS = {
    "right_elbow": ("right_shoulder", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_wrist"),
    "right_knee": ("right_hip", "right_ankle"),
    "left_knee": ("left_hip", "left_ankle"),
}

TARGET_TO_TARGET_JOINT = {
    "right_elbow": "right_elbow",
    "left_elbow": "left_elbow",
    "right_knee": "right_knee",
    "left_knee": "left_knee",
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


def crop_and_resize(rgb, x1, y1, x2, y2, input_size):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop")
    crop = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return crop


def image_to_tensor(rgb):
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0


def full_to_crop_norm(x, y, x1, y1, x2, y2):
    cw = max(float(x2) - float(x1), 1.0)
    ch = max(float(y2) - float(y1), 1.0)
    xn = (safe_float(x) - float(x1)) / cw
    yn = (safe_float(y) - float(y1)) / ch
    return float(xn), float(yn)


def crop_norm_to_full(xn, yn, x1, y1, x2, y2):
    x = float(x1) + float(xn) * max(float(x2) - float(x1), 1.0)
    y = float(y1) + float(yn) * max(float(y2) - float(y1), 1.0)
    return x, y


def get_joint_xy(row, joint_name):
    j = MPII_ID[joint_name]
    x = safe_float(row.get(f"j{j}_orig_x", np.nan))
    y = safe_float(row.get(f"j{j}_orig_y", np.nan))
    return x, y


def get_gt_triplet(row):
    target_name = row["target_name"]
    a_name, b_name = TARGET_TO_ANCHORS[target_name]
    t_name = TARGET_TO_TARGET_JOINT[target_name]

    ax, ay = get_joint_xy(row, a_name)
    bx, by = get_joint_xy(row, b_name)
    tx, ty = get_joint_xy(row, t_name)

    # Use original target columns as fallback.
    if not np.isfinite(tx) or not np.isfinite(ty):
        tx = safe_float(row.get("target_orig_x", np.nan))
        ty = safe_float(row.get("target_orig_y", np.nan))

    return a_name, b_name, ax, ay, bx, by, tx, ty


def merge_crop_and_annotations(crop_csv, annotation_csv):
    crop = pd.read_csv(crop_csv)
    ann = pd.read_csv(annotation_csv)

    keep = ["occluded_image_name", "target_name"]
    for j in range(16):
        for suffix in ["orig_x", "orig_y", "mask"]:
            c = f"j{j}_{suffix}"
            if c in ann.columns:
                keep.append(c)

    keep = [c for c in keep if c in ann.columns]
    ann = ann[keep].drop_duplicates(subset=["occluded_image_name", "target_name"]).copy()

    out = crop.merge(ann, on=["occluded_image_name", "target_name"], how="left")
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


def filter_valid_gt_inside_crop(df, margin=0.00):
    rows = []
    for _, row in df.iterrows():
        try:
            _, _, ax, ay, bx, by, tx, ty = get_gt_triplet(row)
            pts = [(ax, ay), (bx, by), (tx, ty)]

            ok = True
            for x, y in pts:
                if not np.isfinite(x) or not np.isfinite(y):
                    ok = False
                    break

                xn, yn = full_to_crop_norm(
                    x, y,
                    row["crop_x1"], row["crop_y1"], row["crop_x2"], row["crop_y2"]
                )
                if xn < -margin or xn > 1 + margin or yn < -margin or yn > 1 + margin:
                    ok = False
                    break

            if ok:
                rows.append(row)
        except Exception:
            continue

    if not rows:
        return df.reset_index(drop=True)

    return pd.DataFrame(rows).reset_index(drop=True)
