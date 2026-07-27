import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.anchor_dataset_utils import ANCHOR_NAMES, crop_and_resize, decode_heatmaps, crop_norm_to_image, load_json
from src.anchor_net_model import AnchorNet


def make_gt_bbox_from_row(row, image_w, image_h, margin_ratio=0.25):
    xs = []
    ys = []
    for j in range(16):
        x = float(row.get(f"j{j}_orig_x", 0.0))
        y = float(row.get(f"j{j}_orig_y", 0.0))
        if x > 0 and y > 0:
            xs.append(x)
            ys.append(y)

    if len(xs) < 3:
        return [0, 0, image_w - 1, image_h - 1]

    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    side = max(x2 - x1, y2 - y1, 20.0)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = side * (1 + 2 * margin_ratio)
    return [max(0, cx - side / 2), max(0, cy - side / 2), min(image_w - 1, cx + side / 2), min(image_h - 1, cy + side / 2)]


class SynthAnchorPredDataset(Dataset):
    def __init__(self, df, image_dir, img_size=256, crop_mode="gt_bbox"):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.img_size = int(img_size)
        self.crop_mode = crop_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row["occluded_image_name"]
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        if self.crop_mode == "full":
            crop_box = [0, 0, w - 1, h - 1]
        else:
            crop_box = make_gt_bbox_from_row(row, w, h)

        crop, crop_box = crop_and_resize(img, crop_box, self.img_size)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(crop_box, dtype=torch.float32), idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage4_anchor_net_predictions")
    parser.add_argument("--crop_mode", type=str, default="gt_bbox", choices=["gt_bbox", "full"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(Path(args.model_dir) / "model_config.json")
    img_size = int(config["img_size"])
    base = int(config["base"])
    num_anchors = int(config["num_anchors"])

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AnchorNet(num_anchors=num_anchors, base=base).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "best_model.pt", map_location=device))
    model.eval()

    df = pd.read_csv(args.metadata_csv)
    ds = SynthAnchorPredDataset(df, args.image_dir, img_size=img_size, crop_mode=args.crop_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []

    with torch.no_grad():
        for img, crop_box, idx in tqdm(loader, desc="Predicting AnchorNet anchors"):
            img = img.to(device)
            pred_heat, pred_vis = model(img)
            heat_np = torch.sigmoid(pred_heat).cpu().numpy()
            vis_np = torch.sigmoid(pred_vis).cpu().numpy()
            crop_np = crop_box.cpu().numpy()

            for bi in range(img.size(0)):
                row = df.iloc[int(idx[bi])].to_dict()
                coords, hm_scores = decode_heatmaps(heat_np[bi])
                cb = crop_np[bi]

                row["anchor_crop_x1"] = float(cb[0])
                row["anchor_crop_y1"] = float(cb[1])
                row["anchor_crop_x2"] = float(cb[2])
                row["anchor_crop_y2"] = float(cb[3])
                row["anchor_crop_mode"] = args.crop_mode

                for j, name in enumerate(ANCHOR_NAMES):
                    x, y = crop_norm_to_image(coords[j, 0], coords[j, 1], cb)
                    conf = float(vis_np[bi, j] * hm_scores[j])
                    row[f"anchor_{name}_x"] = x
                    row[f"anchor_{name}_y"] = y
                    row[f"anchor_{name}_conf"] = conf
                rows.append(row)

    out_df = pd.DataFrame(rows)

    # compute anchor prediction errors when GT coordinates exist in input
    for name in ANCHOR_NAMES:
        gt_x = None
        gt_y = None
        # stage2B metadata may not have named anchor GT, but j ids are present.
        # Create rough mapping.
        mpii_map = {
            "right_shoulder": 12,
            "right_wrist": 10,
            "right_hip": 2,
            "right_ankle": 0,
            "left_shoulder": 13,
            "left_wrist": 15,
            "left_hip": 3,
            "left_ankle": 5,
        }
        jid = mpii_map[name]
        if f"j{jid}_orig_x" in out_df.columns:
            out_df[f"anchor_{name}_err"] = np.sqrt(
                (out_df[f"anchor_{name}_x"] - out_df[f"j{jid}_orig_x"]) ** 2
                + (out_df[f"anchor_{name}_y"] - out_df[f"j{jid}_orig_y"]) ** 2
            )

    err_cols = [c for c in out_df.columns if c.endswith("_err") and c.startswith("anchor_")]
    if err_cols:
        out_df["mean_anchor_err"] = out_df[err_cols].mean(axis=1)

    out_csv = out_dir / "anchor_net_predictions.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    if "mean_anchor_err" in out_df.columns:
        print(out_df["mean_anchor_err"].describe())


if __name__ == "__main__":
    main()
