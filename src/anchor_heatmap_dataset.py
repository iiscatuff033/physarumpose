import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .heatmap_utils import (
    ANCHOR_NAMES,
    FLIP_PAIRS,
    crop_and_resize,
    expand_bbox,
    image_to_tensor,
    make_heatmaps,
    orig_to_crop_xy,
)


class AnchorHeatmapDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        input_w=288,
        input_h=384,
        hm_w=72,
        hm_h=96,
        sigma=2.0,
        train=False,
        bbox_margin=0.20,
        cutout_prob=0.35,
        flip_prob=0.50,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.hm_w = int(hm_w)
        self.hm_h = int(hm_h)
        self.sigma = float(sigma)
        self.train = train
        self.bbox_margin = float(bbox_margin)
        self.cutout_prob = float(cutout_prob)
        self.flip_prob = float(flip_prob)

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_name):
        p = self.image_dir / str(image_name)
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f'Could not read image: {p}')
        return img

    def _get_box(self, row):
        x1 = float(row['bbox_x1'])
        y1 = float(row['bbox_y1'])
        x2 = float(row['bbox_x2'])
        y2 = float(row['bbox_y2'])
        return expand_bbox(x1, y1, x2, y2, margin=self.bbox_margin)

    def _points_from_row(self, row, meta):
        pts = []
        vis = []
        for name in ANCHOR_NAMES:
            x = float(row[f'{name}_x'])
            y = float(row[f'{name}_y'])
            v = float(row[f'{name}_vis'])
            cx, cy = orig_to_crop_xy(x, y, meta)
            pts.append([cx, cy])
            vis.append(v)
        return np.array(pts, dtype=np.float32), np.array(vis, dtype=np.float32)

    def _color_aug(self, img):
        if not self.train:
            return img
        # simple brightness/contrast on uint8 image
        if random.random() < 0.50:
            alpha = random.uniform(0.75, 1.25)
            beta = random.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img

    def _cutout(self, img):
        if not self.train or random.random() > self.cutout_prob:
            return img
        h, w = img.shape[:2]
        rw = random.randint(max(8, w // 12), max(10, w // 4))
        rh = random.randint(max(8, h // 12), max(10, h // 4))
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        x1 = max(0, cx - rw // 2)
        y1 = max(0, cy - rh // 2)
        x2 = min(w, cx + rw // 2)
        y2 = min(h, cy + rh // 2)
        value = random.choice([0, 80, 127, 200])
        img[y1:y2, x1:x2] = value
        return img

    def _flip(self, img, pts, vis):
        if not self.train or random.random() > self.flip_prob:
            return img, pts, vis
        img = cv2.flip(img, 1)
        pts = pts.copy()
        vis = vis.copy()
        pts[:, 0] = self.input_w - 1 - pts[:, 0]

        new_pts = pts.copy()
        new_vis = vis.copy()
        for i, j in FLIP_PAIRS.items():
            new_pts[i] = pts[j]
            new_vis[i] = vis[j]
        return img, new_pts, new_vis

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row['image_name'])
        box = self._get_box(row)
        crop, meta = crop_and_resize(img, box, self.input_w, self.input_h)
        pts, vis = self._points_from_row(row, meta)

        crop, pts, vis = self._flip(crop, pts, vis)
        crop = self._color_aug(crop)
        crop = self._cutout(crop)

        heatmaps, masks = make_heatmaps(
            pts,
            vis,
            input_w=self.input_w,
            input_h=self.input_h,
            hm_w=self.hm_w,
            hm_h=self.hm_h,
            sigma=self.sigma,
        )

        x = image_to_tensor(crop)
        y = torch.tensor(heatmaps, dtype=torch.float32)
        m = torch.tensor(masks, dtype=torch.float32)

        return x, y, m, idx
