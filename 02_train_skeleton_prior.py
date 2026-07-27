import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model import SkeletonPriorMLP
from src.utils import TASKS, build_feature, build_target, save_json, set_seed


class LimbTaskDataset(Dataset):
    def __init__(self, df, num_tasks):
        self.df = df.reset_index(drop=True)
        self.num_tasks = num_tasks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = build_feature(row, self.num_tasks)
        y = build_target(row)
        task_id = int(row["task_id"])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), task_id


def split_dataframe(df, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)

    image_names = df["image_name"].dropna().unique()
    rng.shuffle(image_names)

    n_val = max(1, int(len(image_names) * val_ratio))
    val_images = set(image_names[:n_val])

    val_df = df[df["image_name"].isin(val_images)].reset_index(drop=True)
    train_df = df[~df["image_name"].isin(val_images)].reset_index(drop=True)

    return train_df, val_df


def compute_metrics(pred, target):
    err = torch.linalg.norm(pred - target, dim=1)
    return {
        "mean_l2": float(err.mean().item()),
        "median_l2": float(err.median().item()),
    }


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_pred = []
    all_target = []

    loss_fn = torch.nn.SmoothL1Loss()

    for x, y, _task_id in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = loss_fn(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        all_pred.append(pred.detach().cpu())
        all_target.append(y.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    metrics = compute_metrics(all_pred, all_target)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/skeleton_prior_mpii")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if len(df) == 0:
        raise ValueError("CSV has 0 rows. Check MPII parsing and visibility filtering.")

    num_tasks = len(TASKS)
    train_df, val_df = split_dataframe(df, val_ratio=args.val_ratio, seed=args.seed)

    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print("\nTrain task distribution:")
    print(train_df["task_name"].value_counts())

    train_loader = DataLoader(
        LimbTaskDataset(train_df, num_tasks=num_tasks),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        LimbTaskDataset(val_df, num_tasks=num_tasks),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    model = SkeletonPriorMLP(
        num_tasks=num_tasks,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mean_l2": train_metrics["mean_l2"],
            "val_loss": val_metrics["loss"],
            "val_mean_l2": val_metrics["mean_l2"],
            "val_median_l2": val_metrics["median_l2"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.5f} | "
            f"val_loss={row['val_loss']:.5f} | "
            f"val_mean_l2={row['val_mean_l2']:.5f}"
        )

        if val_metrics["mean_l2"] < best_val:
            best_val = val_metrics["mean_l2"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            save_json({
                "num_tasks": num_tasks,
                "tasks": TASKS,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "best_val_mean_l2": best_val,
            }, out_dir / "model_config.json")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    print("\nTraining complete.")
    print(f"Best val mean L2: {best_val:.5f}")
    print(f"Saved model to: {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
