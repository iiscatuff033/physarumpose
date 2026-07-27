from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .pred_region_common import make_region_input, make_target_heatmap


class TargetRegionDataset(Dataset):
    def __init__(self, df, image_dir, input_size=256, sigma=5, augment=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.input_size = input_size
        self.sigma = sigma
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row["occluded_image_name"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            image_bgr = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        x = make_region_input(row, image_bgr, size=self.input_size, sigma=self.sigma)
        y = make_target_heatmap(row, image_bgr, size=self.input_size, sigma=self.sigma)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), idx
