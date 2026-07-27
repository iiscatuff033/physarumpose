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


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def find_teacher_anchor(row, name):
    patterns = [
        (f"anchor_pred_{name}_x", f"anchor_pred_{name}_y", f"anchor_pred_{name}_conf"),
        (f"strong_anchor_{name}_x", f"strong_anchor_{name}_y", f"strong_anchor_{name}_conf"),
        (f"pred_{name}_x", f"pred_{name}_y", f"pred_{name}_conf"),
        (f"anchor_{name}_x", f"anchor_{name}_y", f"anchor_{name}_conf"),
        (f"{name}_x", f"{name}_y", f"{name}_conf"),
    ]

    for xk, yk, ck in patterns:
        if xk in row.index and yk in row.index:
            x = safe_float(row[xk])
            y = safe_float(row[yk])
            c = safe_float(row[ck], 1.0) if ck in row.index else 1.0
            return x, y, c

    return np.nan, np.nan, 0.0


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


class TeacherAnchorSoftSlimeDataset(Dataset):
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
        anchor_noise_px=0.0,
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.augment = augment
        self.heatmap_sigma = float(heatmap_sigma)
        self.anchor_noise_px = float(anchor_noise_px)

        if split_df is not None:
            df = split_df.copy()
        else:
            df = pd.read_csv(self.csv_path)

        if max_samples is not None and max_samples > 0:
            df = df.head(max_samples).reset_index(drop=True)

        df = df[df["target_name"].isin(TARGET_TO_ID.keys())].copy()
        df = df[np.isfinite(df["target_orig_x"]) & np.isfinite(df["target_orig_y"])].copy()

        # Filter rows where required teacher anchors exist.
        good_rows = []
        for _, row in df.iterrows():
            ok = True
            for name in ANCHOR_NAMES:
                x, y, c = find_teacher_anchor(row, name)
                if not np.isfinite(x) or not np.isfinite(y) or x <= 0 or y <= 0:
                    ok = False
                    break
            if ok:
                good_rows.append(row)

        if good_rows:
            self.df = pd.DataFrame(good_rows).reset_index(drop=True)
        else:
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

        if self.augment and np.random.rand() < 0.5:
            factor = np.random.uniform(0.85, 1.15)
            img_resized = np.clip(img_resized.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        return img_t, w, h

    def _get_teacher_anchors(self, row, img_w, img_h):
        xy = np.zeros((len(ANCHOR_NAMES), 2), dtype=np.float32)
        conf = np.zeros((len(ANCHOR_NAMES),), dtype=np.float32)

        for i, name in enumerate(ANCHOR_NAMES):
            x, y, c = find_teacher_anchor(row, name)

            if self.augment and self.anchor_noise_px > 0:
                x += np.random.normal(0.0, self.anchor_noise_px)
                y += np.random.normal(0.0, self.anchor_noise_px)

            if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
                xy[i, 0] = np.clip(x / max(img_w, 1), 0.0, 1.0)
                xy[i, 1] = np.clip(y / max(img_h, 1), 0.0, 1.0)
                conf[i] = np.clip(c, 0.0, 1.0)
            else:
                xy[i, :] = 0.5
                conf[i] = 0.0

        return xy, conf

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

        teacher_anchor_xy, teacher_anchor_conf = self._get_teacher_anchors(row, img_w, img_h)

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

        for col in [
            "yolo_target_x", "yolo_target_y", "yolo_target_error_px",
            "prior_pred_x", "prior_pred_y", "prior_error_px",
            "geometry_x", "geometry_y", "geometry_error_px",
            "mean_anchor_err",
        ]:
            if col in row.index:
                meta[col] = safe_float(row.get(col, np.nan))

        # Save teacher anchors in pixel coords for eval/visualization.
        for i, name in enumerate(ANCHOR_NAMES):
            meta[f"teacher_{name}_x"] = float(teacher_anchor_xy[i, 0] * img_w)
            meta[f"teacher_{name}_y"] = float(teacher_anchor_xy[i, 1] * img_h)
            meta[f"teacher_{name}_conf"] = float(teacher_anchor_conf[i])

        return {
            "image": img_t,
            "teacher_anchor_xy": torch.tensor(teacher_anchor_xy, dtype=torch.float32),
            "teacher_anchor_conf": torch.tensor(teacher_anchor_conf, dtype=torch.float32),
            "target_xy": torch.tensor(target_xy, dtype=torch.float32),
            "target_id": torch.tensor(target_id, dtype=torch.long),
            "target_heatmaps": torch.tensor(np.stack(target_heatmaps), dtype=torch.float32),
            "meta": meta,
        }
