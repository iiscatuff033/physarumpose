import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.pred_region_common import make_region_input, make_target_heatmap, split_by_image, save_json, heatmap_peak_to_orig
from src.target_region_model import TargetRegionNet


from src.target_region_dataset import TargetRegionDataset

def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total = 0.0
    with torch.set_grad_enabled(train):
        for x, y, _idx in tqdm(loader, leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def predict_df(model, df, image_dir, input_size, sigma, device, batch_size=32):
    ds = TargetRegionDataset(df, image_dir, input_size=input_size, sigma=sigma, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = df.copy().reset_index(drop=True)
    pred_x = np.zeros(len(out), dtype=np.float32)
    pred_y = np.zeros(len(out), dtype=np.float32)
    pred_conf = np.zeros(len(out), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for x, _y, idx in tqdm(loader, leave=False):
            x = x.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy()
            for j, row_idx in enumerate(idx.numpy()):
                row = out.iloc[row_idx]
                img = cv2.imread(str(Path(image_dir) / row["occluded_image_name"]))
                if img is None:
                    continue
                h, w = img.shape[:2]
                ox, oy, conf = heatmap_peak_to_orig(prob[j, 0], w, h, input_size)
                pred_x[row_idx] = ox
                pred_y[row_idx] = oy
                pred_conf[row_idx] = conf
    out["region_pred_x"] = pred_x
    out["region_pred_y"] = pred_y
    out["region_pred_conf"] = pred_conf
    out["region_error_px"] = np.sqrt((out["region_pred_x"] - out["target_orig_x"]) ** 2 + (out["region_pred_y"] - out["target_orig_y"]) ** 2)
    return out


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5c_target_region_net")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["occluded_image_name", "target_name", "target_orig_x", "target_orig_y"]).reset_index(drop=True)
    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Total samples: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    train_ds = TargetRegionDataset(train_df, args.image_dir, args.input_size, args.sigma, augment=False)
    val_ds = TargetRegionDataset(val_df, args.image_dir, args.input_size, args.sigma, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = TargetRegionNet(in_ch=10, base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, train=True)
        va = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {"epoch": epoch, "train_loss": tr, "val_loss": va}
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={tr:.6f} | val_loss={va:.6f}")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            save_json({"input_size": args.input_size, "sigma": args.sigma, "base": args.base, "best_val_loss": best_val}, out_dir / "model_config.json")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    val_pred = predict_df(model, val_df, args.image_dir, args.input_size, args.sigma, device, batch_size=args.batch_size)
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)
    summary = val_pred.groupby("target_name").agg(count=("region_error_px", "count"), mean_region_error_px=("region_error_px", "mean"), median_region_error_px=("region_error_px", "median"), mean_conf=("region_pred_conf", "mean")).reset_index()
    overall = pd.DataFrame([{"target_name": "OVERALL", "count": len(val_pred), "mean_region_error_px": val_pred["region_error_px"].mean(), "median_region_error_px": val_pred["region_error_px"].median(), "mean_conf": val_pred["region_pred_conf"].mean()}])
    summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(out_dir / "val_region_summary.csv", index=False)
    print("\nValidation region summary:")
    print(summary)


if __name__ == "__main__":
    main()
