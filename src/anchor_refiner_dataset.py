from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .stage5h_common import (
    TARGET_TO_ID,
    make_heatmap,
    crop_and_resize,
    image_to_tensor,
    get_gt_anchor_pair,
    get_gt_target,
    safe_float,
)


class InstanceAnchorRefinerDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        input_size=256,
        heatmap_size=64,
        augment=False,
        heatmap_sigma=2.0,
        skip_bad_gt=True,
    ):
        self.image_dir = Path(image_dir)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.augment = augment
        self.heatmap_sigma = float(heatmap_sigma)

        rows = []
        for _, row in df.iterrows():
            try:
                _, _, ax, ay, bx, by = get_gt_anchor_pair(row)
                tx, ty = get_gt_target(row)

                vals = [ax, ay, bx, by, tx, ty]
                if skip_bad_gt and any(not np.isfinite(v) for v in vals):
                    continue
                rows.append(row)
            except Exception:
                continue

        self.df = df.__class__(rows).reset_index(drop=True) if rows else df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_crop(self, row):
        image_path = self.image_dir / str(row["occluded_image_name"])
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        x1, y1, x2, y2 = [int(row[c]) for c in ["crop_x1", "crop_y1", "crop_x2", "crop_y2"]]
        crop = crop_and_resize(rgb, x1, y1, x2, y2, self.input_size)

        if self.augment:
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.85, 1.15)
                crop = np.clip(crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)

            if np.random.rand() < 0.25:
                noise = np.random.normal(0, 4, crop.shape).astype(np.float32)
                crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return crop, x1, y1, x2, y2

    def _full_to_crop_norm(self, x, y, x1, y1, x2, y2):
        cw = max(x2 - x1, 1)
        ch = max(y2 - y1, 1)
        xn = (safe_float(x) - x1) / cw
        yn = (safe_float(y) - y1) / ch
        return float(np.clip(xn, 0.0, 1.0)), float(np.clip(yn, 0.0, 1.0))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop, x1, y1, x2, y2 = self._load_crop(row)

        a_name, b_name, ax, ay, bx, by = get_gt_anchor_pair(row)
        tx, ty = get_gt_target(row)

        a_xy = self._full_to_crop_norm(ax, ay, x1, y1, x2, y2)
        b_xy = self._full_to_crop_norm(bx, by, x1, y1, x2, y2)
        t_xy = self._full_to_crop_norm(tx, ty, x1, y1, x2, y2)

        xy = np.array([a_xy, b_xy, t_xy], dtype=np.float32)

        heatmaps = []
        for x, y in xy:
            heatmaps.append(make_heatmap(x, y, self.heatmap_size, self.heatmap_size, sigma=self.heatmap_sigma))

        meta = {
            "sample_id": int(row["sample_id"]),
            "image_name": str(row.get("image_name", "")),
            "occluded_image_name": str(row["occluded_image_name"]),
            "target_name": str(row["target_name"]),
            "anchor_a_name": a_name,
            "anchor_b_name": b_name,
            "crop_x1": float(x1),
            "crop_y1": float(y1),
            "crop_x2": float(x2),
            "crop_y2": float(y2),
            "target_orig_x": float(tx),
            "target_orig_y": float(ty),
            "gt_anchor_a_x": float(ax),
            "gt_anchor_a_y": float(ay),
            "gt_anchor_b_x": float(bx),
            "gt_anchor_b_y": float(by),
            "gated_error_px": float(row.get("gated_error_px", np.nan)),
            "suspicious_case": int(row.get("suspicious_case", 0)),
        }

        return {
            "image": image_to_tensor(crop),
            "xy": torch.tensor(xy, dtype=torch.float32),
            "heatmaps": torch.tensor(np.stack(heatmaps), dtype=torch.float32),
            "target_id": torch.tensor(TARGET_TO_ID[str(row["target_name"])], dtype=torch.long),
            "meta": meta,
        }
