from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_TO_ID = {
    "right_elbow": 0,
    "left_elbow": 1,
    "right_knee": 2,
    "left_knee": 3,
}

ID_TO_TARGET = {v: k for k, v in TARGET_TO_ID.items()}

# Anchor order used by the model
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

# MPII indexing convention:
# 0 r ankle, 1 r knee, 2 r hip, 3 l hip, 4 l knee, 5 l ankle,
# 6 pelvis, 7 thorax, 8 upper neck, 9 head top,
# 10 r wrist, 11 r elbow, 12 r shoulder,
# 13 l shoulder, 14 l elbow, 15 l wrist
ANCHOR_TO_MPII = {
    "right_shoulder": 12,
    "right_wrist": 10,
    "right_hip": 2,
    "right_ankle": 0,
    "left_shoulder": 13,
    "left_wrist": 15,
    "left_hip": 3,
    "left_ankle": 5,
}


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def make_gaussian_heatmap(x_norm, y_norm, h, w, sigma=2.0):
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


class E2ESoftSlimeDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        input_size=256,
        heatmap_size=64,
        max_samples=-1,
        split_df=None,
        augment=False,
        heatmap_sigma=2.0,
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.augment = augment
        self.heatmap_sigma = float(heatmap_sigma)

        if split_df is not None:
            df = split_df.copy()
        else:
            df = pd.read_csv(self.csv_path)

        if max_samples is not None and max_samples > 0:
            df = df.head(max_samples).reset_index(drop=True)

        # Keep rows with target name and target coordinate.
        df = df[df["target_name"].isin(TARGET_TO_ID.keys())].copy()
        df = df[np.isfinite(df["target_orig_x"]) & np.isfinite(df["target_orig_y"])].copy()
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_name):
        path = self.image_dir / str(image_name)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            # simple brightness jitter
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.85, 1.15)
                img_resized = np.clip(img_resized.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        return img_t, w, h

    def _get_anchor_gt(self, row, img_w, img_h):
        xy = np.zeros((len(ANCHOR_NAMES), 2), dtype=np.float32)
        mask = np.zeros((len(ANCHOR_NAMES),), dtype=np.float32)

        for i, name in enumerate(ANCHOR_NAMES):
            j = ANCHOR_TO_MPII[name]

            x = safe_float(row.get(f"j{j}_orig_x", np.nan))
            y = safe_float(row.get(f"j{j}_orig_y", np.nan))
            m = safe_float(row.get(f"j{j}_mask", 1.0), 1.0)

            # Fallback to already predicted anchor if GT not present.
            if not np.isfinite(x) or not np.isfinite(y):
                x = safe_float(row.get(f"anchor_pred_{name}_x", np.nan))
                y = safe_float(row.get(f"anchor_pred_{name}_y", np.nan))
                m = 1.0 if np.isfinite(x) and np.isfinite(y) else 0.0

            if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0 and m > 0:
                xy[i, 0] = np.clip(x / max(img_w, 1), 0.0, 1.0)
                xy[i, 1] = np.clip(y / max(img_h, 1), 0.0, 1.0)
                mask[i] = 1.0

        return xy, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row.get("occluded_image_name", row.get("image_name", ""))
        img_t, img_w, img_h = self._load_image(image_name)

        target_name = row["target_name"]
        target_id = TARGET_TO_ID[target_name]

        tx = safe_float(row["target_orig_x"])
        ty = safe_float(row["target_orig_y"])
        target_xy = np.array([
            np.clip(tx / max(img_w, 1), 0.0, 1.0),
            np.clip(ty / max(img_h, 1), 0.0, 1.0),
        ], dtype=np.float32)

        anchor_xy, anchor_mask = self._get_anchor_gt(row, img_w, img_h)

        # Heatmaps: 8 anchors + 4 target heatmaps
        anchor_heatmaps = []
        for i in range(len(ANCHOR_NAMES)):
            if anchor_mask[i] > 0:
                hm = make_gaussian_heatmap(
                    anchor_xy[i, 0], anchor_xy[i, 1],
                    self.heatmap_size, self.heatmap_size,
                    sigma=self.heatmap_sigma,
                )
            else:
                hm = np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32)
            anchor_heatmaps.append(hm)

        target_heatmaps = []
        for t in range(4):
            if t == target_id:
                hm = make_gaussian_heatmap(
                    target_xy[0], target_xy[1],
                    self.heatmap_size, self.heatmap_size,
                    sigma=self.heatmap_sigma,
                )
            else:
                hm = np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32)
            target_heatmaps.append(hm)

        meta = {
            "index": int(idx),
            "image_name": str(row.get("image_name", "")),
            "occluded_image_name": str(image_name),
            "target_name": str(target_name),
            "img_w": float(img_w),
            "img_h": float(img_h),
            "target_orig_x": float(tx),
            "target_orig_y": float(ty),
        }

        # Baselines if present
        for col in [
            "yolo_target_x", "yolo_target_y", "yolo_target_error_px",
            "region_pred_x", "region_pred_y", "region_error_px",
            "geometry_x", "geometry_y", "geometry_error_px",
        ]:
            if col in row.index:
                meta[col] = safe_float(row.get(col, np.nan))

        return {
            "image": img_t,
            "anchor_xy": torch.tensor(anchor_xy, dtype=torch.float32),
            "anchor_mask": torch.tensor(anchor_mask, dtype=torch.float32),
            "target_xy": torch.tensor(target_xy, dtype=torch.float32),
            "target_id": torch.tensor(target_id, dtype=torch.long),
            "anchor_heatmaps": torch.tensor(np.stack(anchor_heatmaps), dtype=torch.float32),
            "target_heatmaps": torch.tensor(np.stack(target_heatmaps), dtype=torch.float32),
            "meta": meta,
        }
