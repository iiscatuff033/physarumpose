import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.anchor_dataset_utils import (
    AnchorDataset,
    ANCHOR_NAMES,
    decode_heatmaps,
    crop_norm_to_image,
    set_seed,
    save_json,
)
from src.anchor_net_model import AnchorNet


def split_by_image(df, val_ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)
    names = df["image_name"].dropna().unique()
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])
    val = df[df["image_name"].isin(val_names)].reset_index(drop=True)
    train = df[~df["image_name"].isin(val_names)].reset_index(drop=True)
    return train, val


def heatmap_loss(pred, target, vis):
    # pred logits -> sigmoid heatmaps
    pred = torch.sigmoid(pred)
    mask = vis[:, :, None, None]
    loss = ((pred - target) ** 2) * mask
    denom = mask.sum() * target.shape[-1] * target.shape[-2] + 1e-6
    return loss.sum() / denom


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    bce = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_hm = 0.0
    total_vis = 0.0

    for img, heat, vis, _coords, _idx in tqdm(loader, leave=False):
        img = img.to(device)
        heat = heat.to(device)
        vis = vis.to(device)

        with torch.set_grad_enabled(train):
            pred_heat, pred_vis = model(img)
            hm_loss = heatmap_loss(pred_heat, heat, vis)
            vis_loss = bce(pred_vis, vis)
            loss = hm_loss + 0.10 * vis_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = img.size(0)
        total_loss += loss.item() * bs
        total_hm += hm_loss.item() * bs
        total_vis += vis_loss.item() * bs

    return {
        "loss": total_loss / len(loader.dataset),
        "heatmap_loss": total_hm / len(loader.dataset),
        "visibility_loss": total_vis / len(loader.dataset),
    }


def predict_val(model, df, image_dir, img_size, heatmap_size, sigma, device):
    ds = AnchorDataset(df, image_dir, img_size=img_size, heatmap_size=heatmap_size, sigma=sigma, train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    rows = []
    model.eval()

    with torch.no_grad():
        for img, _heat, _vis, _coords, idx in tqdm(loader, desc="Predicting val", leave=False):
            img = img.to(device)
            pred_heat, pred_vis = model(img)
            heat_np = torch.sigmoid(pred_heat).cpu().numpy()
            vis_prob = torch.sigmoid(pred_vis).cpu().numpy()

            for bi in range(img.size(0)):
                row = df.iloc[int(idx[bi])].to_dict()
                coords, hm_scores = decode_heatmaps(heat_np[bi])
                crop_box = [row["crop_x1"], row["crop_y1"], row["crop_x2"], row["crop_y2"]]

                errs = []
                for j, name in enumerate(ANCHOR_NAMES):
                    px, py = crop_norm_to_image(coords[j, 0], coords[j, 1], crop_box)
                    conf = float(vis_prob[bi, j] * hm_scores[j])
                    row[f"pred_{name}_x"] = px
                    row[f"pred_{name}_y"] = py
                    row[f"pred_{name}_conf"] = conf

                    if row[f"{name}_vis"] > 0.5:
                        err = float(np.sqrt((px - row[f"{name}_x"]) ** 2 + (py - row[f"{name}_y"]) ** 2))
                        errs.append(err)
                        row[f"pred_{name}_err"] = err
                    else:
                        row[f"pred_{name}_err"] = np.nan

                row["mean_visible_anchor_err"] = float(np.mean(errs)) if errs else np.nan
                rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/anchor_net_mpii")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)

    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    train_ds = AnchorDataset(train_df, args.image_dir, args.img_size, args.heatmap_size, args.sigma, train=True)
    val_ds = AnchorDataset(val_df, args.image_dir, args.img_size, args.heatmap_size, args.sigma, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    model = AnchorNet(num_anchors=len(ANCHOR_NAMES), base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, train=True)
        va = run_epoch(model, val_loader, optimizer, device, train=False)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)

        print(f"Epoch {epoch:03d} | train={tr['loss']:.5f} | val={va['loss']:.5f}")

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            save_json({
                "img_size": args.img_size,
                "heatmap_size": args.heatmap_size,
                "sigma": args.sigma,
                "base": args.base,
                "num_anchors": len(ANCHOR_NAMES),
                "anchor_names": ANCHOR_NAMES,
                "best_val_loss": best_val,
            }, out_dir / "model_config.json")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    pred_df = predict_val(model, val_df, args.image_dir, args.img_size, args.heatmap_size, args.sigma, device)
    pred_df.to_csv(out_dir / "val_predictions.csv", index=False)

    print("\nValidation anchor error:")
    print(pred_df["mean_visible_anchor_err"].describe())
    print(f"\nSaved model to: {out_dir}")


if __name__ == "__main__":
    main()
