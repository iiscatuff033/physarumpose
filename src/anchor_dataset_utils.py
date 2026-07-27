import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


ANCHORS = [
    {"name": "right_shoulder", "mpii_id": 12},
    {"name": "right_wrist", "mpii_id": 10},
    {"name": "right_hip", "mpii_id": 2},
    {"name": "right_ankle", "mpii_id": 0},
    {"name": "left_shoulder", "mpii_id": 13},
    {"name": "left_wrist", "mpii_id": 15},
    {"name": "left_hip", "mpii_id": 3},
    {"name": "left_ankle", "mpii_id": 5},
]

ANCHOR_NAMES = [a["name"] for a in ANCHORS]

# right <-> left swaps for horizontal flip
FLIP_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_and_resize(image_bgr, crop_box, out_size):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = crop_box

    x1 = int(clamp(round(x1), 0, w - 1))
    y1 = int(clamp(round(y1), 0, h - 1))
    x2 = int(clamp(round(x2), x1 + 1, w))
    y2 = int(clamp(round(y2), y1 + 1, h))

    crop = image_bgr[y1:y2, x1:x2].copy()
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    return crop, (x1, y1, x2, y2)


def point_to_crop_norm(x, y, crop_box):
    x1, y1, x2, y2 = crop_box
    cw = max(x2 - x1, 1.0)
    ch = max(y2 - y1, 1.0)
    nx = (x - x1) / cw
    ny = (y - y1) / ch
    return float(nx), float(ny)


def crop_norm_to_image(nx, ny, crop_box):
    x1, y1, x2, y2 = crop_box
    cw = max(x2 - x1, 1.0)
    ch = max(y2 - y1, 1.0)
    x = x1 + nx * cw
    y = y1 + ny * ch
    return float(x), float(y)


def make_gaussian_heatmap(cx, cy, size=64, sigma=2.0):
    yy, xx = np.mgrid[0:size, 0:size]
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return heat.astype(np.float32)


def decode_heatmaps(heatmaps):
    """
    heatmaps: numpy [K,H,W]
    returns coords in normalized crop coordinates and max heat value.
    """
    k, h, w = heatmaps.shape
    coords = []
    scores = []

    for i in range(k):
        hm = heatmaps[i]
        flat_idx = int(np.argmax(hm))
        y, x = divmod(flat_idx, w)
        score = float(hm[y, x])

        nx = (x + 0.5) / w
        ny = (y + 0.5) / h
        coords.append([nx, ny])
        scores.append(score)

    return np.array(coords, dtype=np.float32), np.array(scores, dtype=np.float32)


class AnchorDataset(Dataset):
    def __init__(self, df, image_dir, img_size=256, heatmap_size=64, sigma=2.0, train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.img_size = int(img_size)
        self.heatmap_size = int(heatmap_size)
        self.sigma = float(sigma)
        self.train = bool(train)

    def __len__(self):
        return len(self.df)

    def _load_row(self, row):
        image_path = self.image_dir / row["image_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        crop_box = [row["crop_x1"], row["crop_y1"], row["crop_x2"], row["crop_y2"]]

        # small random crop jitter during training
        if self.train:
            x1, y1, x2, y2 = crop_box
            bw = x2 - x1
            bh = y2 - y1
            jx = np.random.uniform(-0.06, 0.06) * bw
            jy = np.random.uniform(-0.06, 0.06) * bh
            scale = np.random.uniform(0.92, 1.12)
            cx = (x1 + x2) / 2.0 + jx
            cy = (y1 + y2) / 2.0 + jy
            nw = bw * scale
            nh = bh * scale
            crop_box = [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]

        crop, crop_box = crop_and_resize(image, crop_box, self.img_size)

        coords = []
        vis = []

        for name in ANCHOR_NAMES:
            v = float(row[f"{name}_vis"])
            ox = float(row[f"{name}_x"])
            oy = float(row[f"{name}_y"])

            if v > 0.5 and ox > 0 and oy > 0:
                nx, ny = point_to_crop_norm(ox, oy, crop_box)
                if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                    coords.append([nx, ny])
                    vis.append(1.0)
                else:
                    coords.append([0.0, 0.0])
                    vis.append(0.0)
            else:
                coords.append([0.0, 0.0])
                vis.append(0.0)

        coords = np.array(coords, dtype=np.float32)
        vis = np.array(vis, dtype=np.float32)

        if self.train and np.random.rand() < 0.5:
            crop = cv2.flip(crop, 1)
            coords[:, 0] = 1.0 - coords[:, 0]
            for a, b in FLIP_PAIRS:
                coords[[a, b]] = coords[[b, a]]
                vis[[a, b]] = vis[[b, a]]

        # brightness/contrast noise
        if self.train:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-12, 12)
            crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        heatmaps = np.zeros((len(ANCHOR_NAMES), self.heatmap_size, self.heatmap_size), dtype=np.float32)

        for i, (nx, ny) in enumerate(coords):
            if vis[i] > 0.5:
                hx = nx * self.heatmap_size - 0.5
                hy = ny * self.heatmap_size - 0.5
                heatmaps[i] = make_gaussian_heatmap(hx, hy, self.heatmap_size, self.sigma)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        return img, heatmaps, vis, coords

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img, heatmaps, vis, coords = self._load_row(row)
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(heatmaps, dtype=torch.float32),
            torch.tensor(vis, dtype=torch.float32),
            torch.tensor(coords, dtype=torch.float32),
            idx,
        )
