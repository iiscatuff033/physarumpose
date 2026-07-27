import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pred_region_common import load_json, heatmap_peak_to_orig
from src.target_region_model import TargetRegionNet
from src.target_region_dataset import TargetRegionDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5c_target_region_predictions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    config = load_json(model_dir / "model_config.json")
    input_size = int(config["input_size"])
    sigma = float(config["sigma"])
    base = int(config.get("base", 32))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = TargetRegionNet(in_ch=10, base=base).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()
    df = pd.read_csv(args.csv_path).reset_index(drop=True)
    ds = TargetRegionDataset(df, args.image_dir, input_size=input_size, sigma=sigma, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pred_x = np.zeros(len(df), dtype=np.float32)
    pred_y = np.zeros(len(df), dtype=np.float32)
    pred_conf = np.zeros(len(df), dtype=np.float32)
    with torch.no_grad():
        for x, _y, idx in tqdm(loader, desc="Predicting target region"):
            x = x.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy()
            for j, row_idx in enumerate(idx.numpy()):
                row = df.iloc[row_idx]
                img = cv2.imread(str(Path(args.image_dir) / row["occluded_image_name"]))
                if img is None:
                    continue
                h, w = img.shape[:2]
                ox, oy, conf = heatmap_peak_to_orig(prob[j, 0], w, h, input_size)
                pred_x[row_idx] = ox
                pred_y[row_idx] = oy
                pred_conf[row_idx] = conf
    out = df.copy()
    out["region_pred_x"] = pred_x
    out["region_pred_y"] = pred_y
    out["region_pred_conf"] = pred_conf
    out["region_error_px"] = np.sqrt((out["region_pred_x"] - out["target_orig_x"]) ** 2 + (out["region_pred_y"] - out["target_orig_y"]) ** 2)
    out_csv = out_dir / "target_region_predictions.csv"
    out.to_csv(out_csv, index=False)
    summary = out.groupby("target_name").agg(count=("region_error_px", "count"), mean_region_error_px=("region_error_px", "mean"), median_region_error_px=("region_error_px", "median"), mean_conf=("region_pred_conf", "mean")).reset_index()
    overall = pd.DataFrame([{"target_name":"OVERALL","count":len(out),"mean_region_error_px":out["region_error_px"].mean(),"median_region_error_px":out["region_error_px"].median(),"mean_conf":out["region_pred_conf"].mean()}])
    summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(out_dir / "target_region_summary.csv", index=False)
    print("Saved:", out_csv)
    print(summary)


if __name__ == "__main__":
    main()
