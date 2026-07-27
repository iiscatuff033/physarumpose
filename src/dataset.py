from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from .common import TARGET_TO_ID, make_heatmap, crop_and_resize, image_to_tensor, get_gt_triplet, full_to_crop_norm, safe_float

class PoseCropDataset(Dataset):
    def __init__(self, df, image_dir, input_size=256, heatmap_size=64, sigma=2.0, augment=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.sigma = float(sigma)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.df)

    def _load_crop(self, row):
        path = self.image_dir / str(row["occluded_image_name"])
        bgr = cv2.imread(str(path))
        if bgr is None: raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]
        x1,y1,x2,y2 = [int(row[c]) for c in ["crop_x1","crop_y1","crop_x2","crop_y2"]]
        crop = crop_and_resize(rgb, x1,y1,x2,y2, self.input_size)
        if self.augment:
            if np.random.rand() < 0.5:
                crop = np.clip(crop.astype(np.float32)*np.random.uniform(0.85,1.15), 0, 255).astype(np.uint8)
            if np.random.rand() < 0.25:
                crop = np.clip(crop.astype(np.float32)+np.random.normal(0,3,crop.shape), 0, 255).astype(np.uint8)
            if np.random.rand() < 0.12:
                h,w = crop.shape[:2]
                cw, ch = np.random.randint(10, max(12,w//8)), np.random.randint(10, max(12,h//8))
                cx, cy = np.random.randint(0,w), np.random.randint(0,h)
                crop[max(0,cy-ch//2):min(h,cy+ch//2), max(0,cx-cw//2):min(w,cx+cw//2)] = 128
        return crop, img_w, img_h, x1,y1,x2,y2

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop, img_w, img_h, x1,y1,x2,y2 = self._load_crop(row)
        a_name,b_name,ax,ay,bx,by,tx,ty = get_gt_triplet(row)
        a_xy = full_to_crop_norm(ax,ay,x1,y1,x2,y2)
        b_xy = full_to_crop_norm(bx,by,x1,y1,x2,y2)
        t_xy = full_to_crop_norm(tx,ty,x1,y1,x2,y2)
        xy = np.array([[np.clip(a_xy[0],0,1),np.clip(a_xy[1],0,1)],
                       [np.clip(b_xy[0],0,1),np.clip(b_xy[1],0,1)],
                       [np.clip(t_xy[0],0,1),np.clip(t_xy[1],0,1)]], dtype=np.float32)
        heatmaps = np.stack([make_heatmap(x,y,self.heatmap_size,self.heatmap_size,self.sigma) for x,y in xy])
        target_name = str(row["target_name"])
        meta = dict(
            sample_id=int(row.get("sample_id", idx)),
            image_name=str(row.get("image_name", "")),
            occluded_image_name=str(row["occluded_image_name"]),
            target_name=target_name,
            target_id=int(TARGET_TO_ID[target_name]),
            anchor_a_name=str(a_name), anchor_b_name=str(b_name),
            img_w=float(img_w), img_h=float(img_h),
            crop_x1=float(x1), crop_y1=float(y1), crop_x2=float(x2), crop_y2=float(y2),
            gt_anchor_a_x=float(ax), gt_anchor_a_y=float(ay),
            gt_anchor_b_x=float(bx), gt_anchor_b_y=float(by),
            target_orig_x=float(tx), target_orig_y=float(ty),
        )
        for col in ["yolo_target_error_px","gated_error_px","top1_error_px","suspicious_case","good_paper_example"]:
            if col in row.index:
                meta[col] = safe_float(row.get(col), np.nan)
        return {
            "image": image_to_tensor(crop),
            "target_id": torch.tensor(TARGET_TO_ID[target_name], dtype=torch.long),
            "xy": torch.tensor(xy, dtype=torch.float32),
            "heatmaps": torch.tensor(heatmaps, dtype=torch.float32),
            "meta": meta,
        }
