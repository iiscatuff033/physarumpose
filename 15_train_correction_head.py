import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.correction_head_model import CorrectionHeadMLP
from src.common_correction import build_feature, denormalize_point, save_json


class CorrectionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        feats = []
        targets = []
        metas = []

        for _, row in self.df.iterrows():
            f, t, m = build_feature(row)
            feats.append(f)
            targets.append(t)
            metas.append(m)

        self.x = np.stack(feats).astype(np.float32)
        self.y = np.stack(targets).astype(np.float32)
        self.metas = metas

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), idx


def split_by_image(df, val_ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)

    if "image_name" not in df.columns:
        mask = rng.random(len(df)) < val_ratio
        return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)

    names = df["image_name"].dropna().unique()
    rng.shuffle(names)

    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])

    val_df = df[df["image_name"].isin(val_names)].reset_index(drop=True)
    train_df = df[~df["image_name"].isin(val_names)].reset_index(drop=True)
    return train_df, val_df


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()

    loss_fn = torch.nn.SmoothL1Loss()

    total_loss = 0.0
    all_pred = []
    all_target = []

    for x, y, _idx in tqdm(loader, leave=False):
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

    err = torch.linalg.norm(all_pred - all_target, dim=1)

    return {
        "loss": total_loss / len(loader.dataset),
        "mean_norm_error": float(err.mean().item()),
        "median_norm_error": float(err.median().item()),
    }


def predict_dataframe(model, df, device):
    ds = CorrectionDataset(df)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    pred_norm = np.zeros((len(df), 2), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for x, _y, idx in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            pred_norm[idx.numpy()] = pred

    out = df.copy()
    out["corr_norm_x"] = pred_norm[:, 0]
    out["corr_norm_y"] = pred_norm[:, 1]

    pred_x = []
    pred_y = []

    for i, row in out.iterrows():
        _, _, meta = build_feature(row)
        px, py = denormalize_point(row["corr_norm_x"], row["corr_norm_y"], meta["width"], meta["height"])
        pred_x.append(px)
        pred_y.append(py)

    out["corr_pred_x"] = pred_x
    out["corr_pred_y"] = pred_y

    out["corr_error_px"] = np.sqrt(
        (out["corr_pred_x"] - out["target_orig_x"]) ** 2
        + (out["corr_pred_y"] - out["target_orig_y"]) ** 2
    )

    out["corr_better_than_yolo"] = out["corr_error_px"] < out["yolo_target_error_px"]
    out["corr_better_than_prior"] = out["corr_error_px"] < out["prior_error_px"]
    out["corr_better_than_geometry"] = out["corr_error_px"] < out["geometry_error_px"]

    return out


def summarize(df):
    return {
        "count": len(df),
        "yolo_mean_px": df["yolo_target_error_px"].mean(),
        "prior_mean_px": df["prior_error_px"].mean(),
        "geometry_mean_px": df["geometry_error_px"].mean(),
        "corr_mean_px": df["corr_error_px"].mean(),
        "yolo_median_px": df["yolo_target_error_px"].median(),
        "prior_median_px": df["prior_error_px"].median(),
        "geometry_median_px": df["geometry_error_px"].median(),
        "corr_median_px": df["corr_error_px"].median(),
        "corr_win_vs_yolo": df["corr_better_than_yolo"].mean(),
        "corr_win_vs_prior": df["corr_better_than_prior"].mean(),
        "corr_win_vs_geometry": df["corr_better_than_geometry"].mean(),
    }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage2c_correction_head")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    if "yolo_found_person" in df.columns:
        df = df[df["yolo_found_person"] == 1].reset_index(drop=True)

    # Need these fields for training
    required = ["target_orig_x", "target_orig_y", "yolo_target_error_px", "prior_error_px", "geometry_error_px"]
    df = df.dropna(subset=required).reset_index(drop=True)

    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = CorrectionDataset(train_df)
    val_ds = CorrectionDataset(val_df)

    input_dim = train_ds.x.shape[1]

    print(f"Total valid samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Input dim: {input_dim}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CorrectionHeadMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, train=True)
        va = run_epoch(model, val_loader, optimizer, device, train=False)

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_mean_norm_error": tr["mean_norm_error"],
            "val_loss": va["loss"],
            "val_mean_norm_error": va["mean_norm_error"],
            "val_median_norm_error": va["median_norm_error"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.5f} | "
            f"val_loss={row['val_loss']:.5f} | "
            f"val_norm_err={row['val_mean_norm_error']:.5f}"
        )

        if va["mean_norm_error"] < best_val:
            best_val = va["mean_norm_error"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            save_json({
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "best_val_mean_norm_error": best_val,
            }, out_dir / "model_config.json")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    # Load best and predict
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    train_pred = predict_dataframe(model, train_df, device)
    val_pred = predict_dataframe(model, val_df, device)
    all_pred = predict_dataframe(model, df, device)

    train_pred.to_csv(out_dir / "train_predictions.csv", index=False)
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)
    all_pred.to_csv(out_dir / "all_predictions.csv", index=False)

    summary = pd.DataFrame([
        {"split": "train", **summarize(train_pred)},
        {"split": "val", **summarize(val_pred)},
        {"split": "all", **summarize(all_pred)},
    ])
    summary.to_csv(out_dir / "correction_head_summary.csv", index=False)

    per_target = all_pred.groupby("target_name").agg(
        count=("corr_error_px", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        prior_mean_px=("prior_error_px", "mean"),
        geometry_mean_px=("geometry_error_px", "mean"),
        corr_mean_px=("corr_error_px", "mean"),
        yolo_median_px=("yolo_target_error_px", "median"),
        corr_median_px=("corr_error_px", "median"),
        corr_win_vs_yolo=("corr_better_than_yolo", "mean"),
    ).reset_index()
    per_target.to_csv(out_dir / "correction_head_summary_by_target.csv", index=False)

    print("\nSummary:")
    print(summary)

    print("\nBy target:")
    print(per_target)

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
