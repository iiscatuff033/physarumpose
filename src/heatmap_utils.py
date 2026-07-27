import json
from pathlib import Path

import cv2
import numpy as np
import torch


ANCHOR_NAMES = [
    'right_shoulder',
    'right_wrist',
    'right_hip',
    'right_ankle',
    'left_shoulder',
    'left_wrist',
    'left_hip',
    'left_ankle',
]

ANCHOR_TO_MPII = {
    'right_shoulder': 12,
    'right_wrist': 10,
    'right_hip': 2,
    'right_ankle': 0,
    'left_shoulder': 13,
    'left_wrist': 15,
    'left_hip': 3,
    'left_ankle': 5,
}

FLIP_PAIRS = {
    0: 4,  # right_shoulder <-> left_shoulder
    1: 5,  # right_wrist <-> left_wrist
    2: 6,  # right_hip <-> left_hip
    3: 7,  # right_ankle <-> left_ankle
    4: 0,
    5: 1,
    6: 2,
    7: 3,
}


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 + 2:
        x2 = min(w, x1 + 3)
    if y2 <= y1 + 2:
        y2 = min(h, y1 + 3)
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, margin=0.20):
    bw = x2 - x1
    bh = y2 - y1
    return x1 - margin * bw, y1 - margin * bh, x2 + margin * bw, y2 + margin * bh


def crop_and_resize(image_bgr, box, input_w, input_h):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box(*box, w=w, h=h)
    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = image_bgr.copy()
        x1, y1, x2, y2 = 0, 0, w, h
    crop_resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    meta = {
        'crop_x1': float(x1),
        'crop_y1': float(y1),
        'crop_x2': float(x2),
        'crop_y2': float(y2),
        'crop_w': float(max(x2 - x1, 1)),
        'crop_h': float(max(y2 - y1, 1)),
        'input_w': float(input_w),
        'input_h': float(input_h),
    }
    return crop_resized, meta


def orig_to_crop_xy(x, y, meta):
    cx = (float(x) - meta['crop_x1']) / max(meta['crop_w'], 1.0) * meta['input_w']
    cy = (float(y) - meta['crop_y1']) / max(meta['crop_h'], 1.0) * meta['input_h']
    return cx, cy


def crop_to_orig_xy(cx, cy, meta):
    x = float(cx) / max(meta['input_w'], 1.0) * meta['crop_w'] + meta['crop_x1']
    y = float(cy) / max(meta['input_h'], 1.0) * meta['crop_h'] + meta['crop_y1']
    return x, y


def gaussian_heatmap(hm_w, hm_h, cx, cy, sigma=2.0):
    x = np.arange(hm_w, dtype=np.float32)
    y = np.arange(hm_h, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    return heat.astype(np.float32)


def make_heatmaps(points_crop, visible, input_w, input_h, hm_w, hm_h, sigma=2.0):
    num_joints = len(points_crop)
    heatmaps = np.zeros((num_joints, hm_h, hm_w), dtype=np.float32)
    masks = np.zeros((num_joints, 1, 1), dtype=np.float32)

    for j, (x, y) in enumerate(points_crop):
        if visible[j] <= 0:
            continue
        if x < 0 or y < 0 or x >= input_w or y >= input_h:
            continue
        hx = x / max(input_w, 1.0) * hm_w
        hy = y / max(input_h, 1.0) * hm_h
        heatmaps[j] = gaussian_heatmap(hm_w, hm_h, hx, hy, sigma=sigma)
        masks[j, 0, 0] = 1.0

    return heatmaps, masks


def image_to_tensor(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32)


def decode_heatmaps(heatmaps, input_w, input_h, apply_sigmoid=True):
    """
    heatmaps: torch or np array [J,H,W]
    returns crop-space coords and confidence.
    """
    if isinstance(heatmaps, torch.Tensor):
        hm = heatmaps.detach().cpu().float()
        if apply_sigmoid:
            hm = torch.sigmoid(hm)
        hm = hm.numpy()
    else:
        hm = heatmaps.astype(np.float32)
        if apply_sigmoid:
            hm = 1.0 / (1.0 + np.exp(-hm))

    j, hm_h, hm_w = hm.shape
    coords = []
    confs = []

    for k in range(j):
        flat_idx = int(np.argmax(hm[k]))
        y, x = divmod(flat_idx, hm_w)
        conf = float(hm[k, y, x])

        # small sub-pixel refinement using local gradient
        px = float(x)
        py = float(y)
        if 1 <= x < hm_w - 1 and 1 <= y < hm_h - 1:
            dx = hm[k, y, x + 1] - hm[k, y, x - 1]
            dy = hm[k, y + 1, x] - hm[k, y - 1, x]
            px += 0.25 * np.sign(dx)
            py += 0.25 * np.sign(dy)

        crop_x = px / max(hm_w, 1.0) * input_w
        crop_y = py / max(hm_h, 1.0) * input_h
        coords.append([crop_x, crop_y])
        confs.append(conf)

    return np.array(coords, dtype=np.float32), np.array(confs, dtype=np.float32)
