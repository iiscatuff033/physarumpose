import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.stage5h_common import merge_crop_and_strong, split_by_image, save_json
from src.anchor_refiner_dataset import InstanceAnchorRefinerDataset
from src.anchor_refiner_model import InstanceAnchorRefiner, soft_argmax_2d


def collate_fn(batch):
    out = {}
    out["image"] = torch.stack([b["image"] for b in batch], dim=0)
    out["xy"] = torch.stack([b["xy"] for b in batch], dim=0)
    out["heatmaps"] = torch.stack([b["heatmaps"] for b in batch], dim=0)
    out["target_id"] = torch.stack([b["target_id"] for b in batch], dim=0)
    out["meta"] = [b["meta"] for b in batch]
    return out


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if k == "meta":
            out[k] = v
        else:
            out[k] = v.to(device)
    return out


def refiner_loss(logits, batch, w_heat=1.0, w_coord=8.0):
    pred = torch.sigmoid(logits)
    heat_loss = ((pred - batch["heatmaps"]) ** 2).mean()

    xy_pred, conf = soft_argmax_2d(logits, temperature=0.05)
    coord_loss = F.smooth_l1_loss(xy_pred, batch["xy"])

    total = w_heat * heat_loss + w_coord * coord_loss
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_heatmap": float(heat_loss.detach().cpu()),
        "loss_coord": float(coord_loss.detach().cpu()),
    }


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()

    totals = {}
    count = 0

    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)

        with torch.set_grad_enabled(train):
            logits = model(batch["image"])
            loss, parts = refiner_loss(logits, batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        bs = batch["image"].shape[0]
        count += bs
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v * bs

    return {k: v / max(count, 1) for k, v in totals.items()}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_csv", type=str, required=True)
    parser.add_argument("--strong_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5h_instance_anchor_refiner")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merge_crop_and_strong(args.crop_csv, args.strong_csv)
    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)

    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    train_ds = InstanceAnchorRefinerDataset(
        train_df,
        image_dir=args.image_dir,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        augment=True,
    )

    val_ds = InstanceAnchorRefinerDataset(
        val_df,
        image_dir=args.image_dir,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if not args.cpu else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if not args.cpu else False,
    )

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    model = InstanceAnchorRefiner(out_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    save_json(vars(args), out_dir / "config.json")

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_parts = run_epoch(model, train_loader, optimizer, device, train=True)
        val_parts = run_epoch(model, val_loader, optimizer, device, train=False)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_parts.items()})
        row.update({f"val_{k}": v for k, v in val_parts.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={train_parts['loss_total']:.6f} | "
            f"val_total={val_parts['loss_total']:.6f} | "
            f"val_coord={val_parts['loss_coord']:.6f}"
        )

        if val_parts["loss_total"] < best_val:
            best_val = val_parts["loss_total"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            save_json({"best_epoch": epoch, "best_val_loss": best_val}, out_dir / "best_model_info.json")

    print("Training finished.")
    print("Best val:", best_val)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
