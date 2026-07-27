import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.teacher_dataset import TeacherAnchorSoftSlimeDataset, split_by_image
from src.teacher_model import TeacherAnchorSoftSlimeModel
from src.teacher_losses import teacher_anchor_soft_slime_loss
from src.teacher_eval_utils import save_json


def collate_fn(batch):
    out = {}
    keys = ["image", "teacher_anchor_xy", "teacher_anchor_conf", "target_xy", "target_id", "target_heatmaps"]
    for k in keys:
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
            out = model(
                image=batch["image"],
                target_id=batch["target_id"],
                teacher_anchor_xy=batch["teacher_anchor_xy"],
                teacher_anchor_conf=batch["teacher_anchor_conf"],
            )
            loss, parts = teacher_anchor_soft_slime_loss(out, batch, weights)

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
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5fb_teacher_anchor_soft_slime")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--anchor_noise_px", type=float, default=2.0)

    parser.add_argument("--w_target_hm", type=float, default=1.0)
    parser.add_argument("--w_region_coord", type=float, default=2.0)
    parser.add_argument("--w_joint_coord", type=float, default=10.0)
    parser.add_argument("--w_candidate_kl", type=float, default=0.25)
    parser.add_argument("--w_entropy", type=float, default=0.001)
    parser.add_argument("--candidate_sigma", type=float, default=0.06)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    train_ds = TeacherAnchorSoftSlimeDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        split_df=train_df,
        augment=True,
        anchor_noise_px=args.anchor_noise_px,
    )

    val_ds = TeacherAnchorSoftSlimeDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        split_df=val_df,
        augment=False,
        anchor_noise_px=0.0,
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

    model = TeacherAnchorSoftSlimeModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    weights = {
        "target_hm": args.w_target_hm,
        "region_coord": args.w_region_coord,
        "joint_coord": args.w_joint_coord,
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

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={train_parts['loss_total']:.6f} | "
            f"val_total={val_parts['loss_total']:.6f} | "
            f"val_joint={val_parts['loss_joint_coord']:.6f} | "
            f"val_region={val_parts['loss_region_coord']:.6f}"
        )

        if val_parts["loss_total"] < best_val:
            best_val = val_parts["loss_total"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            save_json({"best_epoch": epoch, "best_val_loss": best_val}, out_dir / "best_model_info.json")

        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    print("Training finished.")
    print("Best val loss:", best_val)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
