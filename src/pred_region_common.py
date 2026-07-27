import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TARGET_TO_ANCHORS = {
    "right_elbow": ("right_shoulder", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_wrist"),
    "right_knee": ("right_hip", "right_ankle"),
    "left_knee": ("left_hip", "left_ankle"),
}

TARGET_TO_ID = {
    "right_elbow": 0,
    "left_elbow": 1,
    "right_knee": 2,
    "left_knee": 3,
}

ANCHOR_ORDER = [
    "right_shoulder",
    "right_wrist",
    "right_hip",
    "right_ankle",
    "left_shoulder",
    "left_wrist",
    "left_hip",
    "left_ankle",
]

ANCHOR_INDEX = {name: i for i, name in enumerate(ANCHOR_ORDER)}


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def point_error(px, py, tx, ty):
    if np.isnan(px) or np.isnan(py):
        return np.nan
    return float(np.sqrt((px - tx) ** 2 + (py - ty) ** 2))


def one_hot(index, size):
    arr = np.zeros(size, dtype=np.float32)
    arr[index] = 1.0
    return arr


def find_xy_conf_columns(row, anchor_name):
    idx = ANCHOR_INDEX[anchor_name]

    patterns = [
        (f"{anchor_name}_x", f"{anchor_name}_y", f"{anchor_name}_conf"),
        (f"{anchor_name}_pred_x", f"{anchor_name}_pred_y", f"{anchor_name}_conf"),
        (f"pred_{anchor_name}_x", f"pred_{anchor_name}_y", f"pred_{anchor_name}_conf"),
        (f"anchor_{anchor_name}_x", f"anchor_{anchor_name}_y", f"anchor_{anchor_name}_conf"),
        (f"anchornet_{anchor_name}_x", f"anchornet_{anchor_name}_y", f"anchornet_{anchor_name}_conf"),
        (f"strong_anchor_{anchor_name}_x", f"strong_anchor_{anchor_name}_y", f"strong_anchor_{anchor_name}_conf"),
        (f"strong_{anchor_name}_x", f"strong_{anchor_name}_y", f"strong_{anchor_name}_conf"),
        (f"anchor{idx}_x", f"anchor{idx}_y", f"anchor{idx}_conf"),
        (f"pred_anchor{idx}_x", f"pred_anchor{idx}_y", f"pred_anchor{idx}_conf"),
        (f"a{idx}_x", f"a{idx}_y", f"a{idx}_conf"),
        (f"pred_a{idx}_x", f"pred_a{idx}_y", f"pred_a{idx}_conf"),
    ]

    for xk, yk, ck in patterns:
        if xk in row.index and yk in row.index:
            conf = row[ck] if ck in row.index else 1.0
            return xk, yk, ck if ck in row.index else None, safe_float(row[xk]), safe_float(row[yk]), safe_float(conf, 1.0)

    lower_cols = {c.lower(): c for c in row.index}
    name = anchor_name.lower()
    xs, ys, cs = [], [], []
    for lc, original in lower_cols.items():
        if name in lc and lc.endswith("_x"):
            xs.append(original)
        if name in lc and lc.endswith("_y"):
            ys.append(original)
        if name in lc and ("conf" in lc or "score" in lc):
            cs.append(original)
    if xs and ys:
        xk, yk = xs[0], ys[0]
        ck = cs[0] if cs else None
        conf = row[ck] if ck else 1.0
        return xk, yk, ck, safe_float(row[xk]), safe_float(row[yk]), safe_float(conf, 1.0)

    return None, None, None, np.nan, np.nan, 0.0


def get_anchor(row, anchor_name):
    _, _, _, x, y, conf = find_xy_conf_columns(row, anchor_name)
    return np.array([x, y], dtype=np.float32), float(conf)


def get_target_anchors(row):
    target = row["target_name"]
    a_name, b_name = TARGET_TO_ANCHORS[target]
    a, ac = get_anchor(row, a_name)
    b, bc = get_anchor(row, b_name)
    return a_name, b_name, a, b, ac, bc


def valid_point(p):
    return np.isfinite(p).all() and p[0] > 0 and p[1] > 0


def resize_image_rgb(image_bgr, size):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.float32) / 255.0


def draw_gaussian_np(h, w, x, y, sigma):
    heat = np.zeros((h, w), dtype=np.float32)
    if not np.isfinite(x) or not np.isfinite(y):
        return heat
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
    return heat.astype(np.float32)


def draw_line_map(h, w, p1, p2, thickness=2):
    img = np.zeros((h, w), dtype=np.float32)
    if not valid_point(p1) or not valid_point(p2):
        return img
    x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
    x2, y2 = int(round(float(p2[0]))), int(round(float(p2[1])))
    cv2.line(img, (x1, y1), (x2, y2), 1.0, thickness=thickness)
    return img


def map_point_to_size(x, y, orig_w, orig_h, size):
    return float(x) / max(float(orig_w), 1.0) * size, float(y) / max(float(orig_h), 1.0) * size


def map_point_from_size(x, y, orig_w, orig_h, size):
    return float(x) / size * float(orig_w), float(y) / size * float(orig_h)


def make_region_input(row, image_bgr, size=256, sigma=5):
    """
    Input channels:
    - RGB image: 3
    - target one-hot constant maps: 4
    - anchor1 gaussian: 1
    - anchor2 gaussian: 1
    - anchor line map: 1
    Total: 10 channels
    """
    h0, w0 = image_bgr.shape[:2]
    rgb = resize_image_rgb(image_bgr, size)
    channels = [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]

    target_name = row["target_name"]
    target_id = TARGET_TO_ID[target_name]
    oh = one_hot(target_id, 4)
    for v in oh:
        channels.append(np.full((size, size), float(v), dtype=np.float32))

    try:
        _an, _bn, a, b, ac, bc = get_target_anchors(row)
    except Exception:
        a = np.array([np.nan, np.nan], dtype=np.float32)
        b = np.array([np.nan, np.nan], dtype=np.float32)
        ac, bc = 0.0, 0.0

    if valid_point(a):
        ax, ay = map_point_to_size(a[0], a[1], w0, h0, size)
    else:
        ax, ay = np.nan, np.nan
    if valid_point(b):
        bx, by = map_point_to_size(b[0], b[1], w0, h0, size)
    else:
        bx, by = np.nan, np.nan

    anchor1 = draw_gaussian_np(size, size, ax, ay, sigma=sigma) * float(ac)
    anchor2 = draw_gaussian_np(size, size, bx, by, sigma=sigma) * float(bc)
    line = draw_line_map(size, size, np.array([ax, ay]), np.array([bx, by]), thickness=2) * max(float(min(ac, bc)), 0.2)

    channels.extend([anchor1, anchor2, line])
    arr = np.stack(channels, axis=0).astype(np.float32)
    return arr


def make_target_heatmap(row, image_bgr, size=256, sigma=5):
    h0, w0 = image_bgr.shape[:2]
    tx = safe_float(row.get("target_orig_x", np.nan))
    ty = safe_float(row.get("target_orig_y", np.nan))
    txs, tys = map_point_to_size(tx, ty, w0, h0, size)
    return draw_gaussian_np(size, size, txs, tys, sigma=sigma)[None, :, :]

def heatmap_peak_to_orig(heatmap, orig_w, orig_h, input_size=None):
    """
    Convert the heatmap peak back to original image coordinates.

    input_size is optional and kept only for backward compatibility with
    40_train_target_region_net.py and 41_predict_target_region_net.py.
    The actual heatmap size is read from heatmap.shape.
    """
    if heatmap.ndim == 3:
        heatmap = heatmap[0]

    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    conf = float(heatmap[y, x])

    ox, oy = map_point_from_size(x, y, orig_w, orig_h, heatmap.shape[0])
    return ox, oy, conf


def split_by_image(df, val_ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)
    key = "image_name" if "image_name" in df.columns else "occluded_image_name"
    names = df[key].dropna().unique()
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])
    val = df[df[key].isin(val_names)].reset_index(drop=True)
    train = df[~df[key].isin(val_names)].reset_index(drop=True)
    return train, val
