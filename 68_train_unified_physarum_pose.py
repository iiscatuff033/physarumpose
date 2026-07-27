import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.stage6_common import (
    merge_crop_and_annotations,
    split_by_image,
    filter_valid_gt_inside_crop,
    save_json,
)
from src.unified_dataset import UnifiedPhysarumDataset
from src.unified_model import UnifiedPhysarumPoseModel
from src.unified_losses import unified_physarum_loss


def collate_fn(batch):
    out = {}
    for k in ["image", "target_id", "xy", "heatmaps"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
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


def run_epoch(model, loader, optimizer, device, weights, train=True):
    model.train() if train else model.eval()
    totals = {}
    count = 0

    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)

        with torch.set_grad_enabled(train):
            out = model(batch["image"], batch["target_id"])
            loss, parts = unified_physarum_loss(out, batch, weights)

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
    parser.add_argument("--annotation_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage6a_unified_physarum_pose")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)

    # Loss weights
    parser.add_argument("--w_heatmap", type=float, default=1.0)
    parser.add_argument("--w_coord_all", type=float, default=2.0)
    parser.add_argument("--w_anchor_coord", type=float, default=4.0)
    parser.add_argument("--w_region_coord", type=float, default=4.0)
    parser.add_argument("--w_soft_joint", type=float, default=12.0)
    parser.add_argument("--w_candidate_kl", type=float, default=0.35)
    parser.add_argument("--w_entropy", type=float, default=0.001)
    parser.add_argument("--candidate_sigma", type=float, default=0.06)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merge_crop_and_annotations(args.crop_csv, args.annotation_csv)
    df = filter_valid_gt_inside_crop(df, margin=0.05)

    if args.max_samples is not None and args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    df.to_csv(out_dir / "merged_valid_dataset.csv", index=False)

    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    train_ds = UnifiedPhysarumDataset(
        train_df,
        image_dir=args.image_dir,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        augment=True,
    )

    val_ds = UnifiedPhysarumDataset(
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
    print("Valid samples:", len(df))
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    model = UnifiedPhysarumPoseModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    weights = {
        "heatmap": args.w_heatmap,
        "coord_all": args.w_coord_all,
        "anchor_coord": args.w_anchor_coord,
        "region_coord": args.w_region_coord,
        "soft_joint": args.w_soft_joint,
        "candidate_kl": args.w_candidate_kl,
        "entropy": args.w_entropy,
        "candidate_sigma": args.candidate_sigma,
    }

    config = vars(args)
    config["weights"] = weights
    config["device"] = device
    save_json(config, out_dir / "config.json")

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_parts = run_epoch(model, train_loader, optimizer, device, weights, train=True)
        val_parts = run_epoch(model, val_loader, optimizer, device, weights, train=False)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_parts.items()})
        row.update({f"val_{k}": v for k, v in val_parts.items()})
        history.append(row)

        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={train_parts['loss_total']:.6f} | "
            f"val_total={val_parts['loss_total']:.6f} | "
            f"val_soft_joint={val_parts['loss_soft_joint']:.6f} | "
            f"val_anchor={val_parts['loss_anchor_coord']:.6f}"
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
