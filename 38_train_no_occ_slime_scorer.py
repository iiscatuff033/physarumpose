import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.no_occ_slime_model import NoOccSlimeScorer
from src.no_occ_slime_utils import (
    load_json,
    save_json,
    split_by_image,
    summarize_best,
)


class CandidateDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols

        x = self.df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
        y = self.df["candidate_quality"].fillna(0.0).values.astype(np.float32)

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            idx,
        )


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()

    loss_fn = torch.nn.MSELoss()

    total = 0.0

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

        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


def predict_candidates(model, df, feature_cols, device, batch_size=8192):
    ds = CandidateDataset(df, feature_cols)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    scores = np.zeros(len(df), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for x, _y, idx in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            scores[idx.numpy()] = pred

    out = df.copy()
    out["learned_score"] = scores
    return out


def pick_best_per_sample(cand_df):
    rows = []

    for sample_id, g in cand_df.groupby("sample_id"):
        g = g.copy()

        learned = g.sort_values("learned_score", ascending=False).iloc[0]
        heuristic = g.sort_values("heuristic_slime_score", ascending=False).iloc[0]
        oracle = g.sort_values("candidate_error_px", ascending=True).iloc[0]

        row = {
            "sample_id": sample_id,
            "image_name": learned.get("image_name", ""),
            "occluded_image_name": learned.get("occluded_image_name", ""),
            "target_name": learned["target_name"],

            "learned_candidate_id": int(learned["candidate_id"]),
            "learned_x": float(learned["candidate_x"]),
            "learned_y": float(learned["candidate_y"]),
            "learned_error_px": float(learned["candidate_error_px"]),
            "learned_score": float(learned["learned_score"]),

            "heuristic_candidate_id": int(heuristic["candidate_id"]),
            "heuristic_x": float(heuristic["candidate_x"]),
            "heuristic_y": float(heuristic["candidate_y"]),
            "heuristic_error_px": float(heuristic["candidate_error_px"]),
            "heuristic_score": float(heuristic["heuristic_slime_score"]),

            "oracle_candidate_id": int(oracle["candidate_id"]),
            "oracle_x": float(oracle["candidate_x"]),
            "oracle_y": float(oracle["candidate_y"]),
            "oracle_error_px": float(oracle["candidate_error_px"]),

            "target_orig_x": float(learned["target_orig_x"]),
            "target_orig_y": float(learned["target_orig_y"]),

            "anchor1_x": float(learned["anchor1_x"]),
            "anchor1_y": float(learned["anchor1_y"]),
            "anchor2_x": float(learned["anchor2_x"]),
            "anchor2_y": float(learned["anchor2_y"]),
            "anchor1_conf": float(learned["anchor1_conf"]),
            "anchor2_conf": float(learned["anchor2_conf"]),
            "anchor_min_conf": float(learned["anchor_min_conf"]),
        }

        for col in [
            "yolo_target_x", "yolo_target_y", "yolo_target_error_px",
            "prior_pred_x", "prior_pred_y", "prior_error_px",
            "geometry_x", "geometry_y", "geometry_error_px",
        ]:
            if col in learned.index:
                row[col] = learned[col]

        rows.append(row)

    out = pd.DataFrame(rows)

    if "yolo_target_error_px" in out.columns:
        out["learned_better_than_yolo"] = out["learned_error_px"] < out["yolo_target_error_px"]
        out["heuristic_better_than_yolo"] = out["heuristic_error_px"] < out["yolo_target_error_px"]

    out["learned_better_than_heuristic"] = out["learned_error_px"] < out["heuristic_error_px"]

    return out


def add_error_bins(best_df):
    if "yolo_target_error_px" not in best_df.columns:
        return pd.DataFrame()

    bins = [0, 20, 40, 60, 80, 100, 150, 999999]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"]

    df = best_df.copy()
    df["yolo_error_bin"] = pd.cut(
        df["yolo_target_error_px"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    return df.groupby("yolo_error_bin", observed=False).agg(
        count=("learned_error_px", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        learned_mean_px=("learned_error_px", "mean"),
        heuristic_mean_px=("heuristic_error_px", "mean"),
        oracle_mean_px=("oracle_error_px", "mean"),
        yolo_median_px=("yolo_target_error_px", "median"),
        learned_median_px=("learned_error_px", "median"),
        learned_win_vs_yolo=("learned_better_than_yolo", "mean"),
    ).reset_index()


def summarize_by_target(best_df):
    agg = {
        "count": ("learned_error_px", "count"),
        "learned_mean_px": ("learned_error_px", "mean"),
        "heuristic_mean_px": ("heuristic_error_px", "mean"),
        "oracle_mean_px": ("oracle_error_px", "mean"),
        "learned_median_px": ("learned_error_px", "median"),
        "heuristic_median_px": ("heuristic_error_px", "median"),
        "learned_win_vs_heuristic": ("learned_better_than_heuristic", "mean"),
    }

    if "yolo_target_error_px" in best_df.columns:
        agg["yolo_mean_px"] = ("yolo_target_error_px", "mean")
        agg["yolo_median_px"] = ("yolo_target_error_px", "median")
        agg["learned_win_vs_yolo"] = ("learned_better_than_yolo", "mean")

    return best_df.groupby("target_name").agg(**agg).reset_index()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_csv", type=str, required=True)
    parser.add_argument("--feature_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5b_no_occ_slime_scorer")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = load_json(args.feature_json)
    cand_df = pd.read_csv(args.candidate_csv)

    # Hard protection against accidental occlusion leakage
    forbidden = [c for c in feature_cols if "occ" in c.lower() or "box" in c.lower()]
    if forbidden:
        raise ValueError(f"Forbidden occlusion-related features found: {forbidden}")

    train_df, val_df = split_by_image(cand_df, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = CandidateDataset(train_df, feature_cols)
    val_ds = CandidateDataset(val_df, feature_cols)

    input_dim = train_ds.x.shape[1]

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Candidate rows: {len(cand_df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Input dim: {input_dim}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = NoOccSlimeScorer(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            save_json({
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "feature_cols": feature_cols,
                "uses_occlusion_box_features": 0,
                "best_val_loss": best_val,
            }, out_dir / "model_config.json")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    train_pred = predict_candidates(model, train_df, feature_cols, device, batch_size=args.batch_size)
    val_pred = predict_candidates(model, val_df, feature_cols, device, batch_size=args.batch_size)
    all_pred = predict_candidates(model, cand_df, feature_cols, device, batch_size=args.batch_size)

    train_pred.to_csv(out_dir / "train_candidate_predictions.csv", index=False)
    val_pred.to_csv(out_dir / "val_candidate_predictions.csv", index=False)
    all_pred.to_csv(out_dir / "all_candidate_predictions.csv", index=False)

    train_best = pick_best_per_sample(train_pred)
    val_best = pick_best_per_sample(val_pred)
    all_best = pick_best_per_sample(all_pred)

    train_best.to_csv(out_dir / "train_best_predictions.csv", index=False)
    val_best.to_csv(out_dir / "val_best_predictions.csv", index=False)
    all_best.to_csv(out_dir / "all_best_predictions.csv", index=False)

    summary = pd.DataFrame([
        {"split": "train", **summarize_best(train_best)},
        {"split": "val", **summarize_best(val_best)},
        {"split": "all", **summarize_best(all_best)},
    ])
    summary["uses_occlusion_box_features"] = 0
    summary.to_csv(out_dir / "no_occ_slime_summary.csv", index=False)

    by_target = summarize_by_target(all_best)
    by_target.to_csv(out_dir / "no_occ_slime_summary_by_target.csv", index=False)

    bins = add_error_bins(all_best)
    if len(bins) > 0:
        bins.to_csv(out_dir / "no_occ_slime_error_bins.csv", index=False)

    print("\nSummary:")
    print(summary)

    print("\nBy target:")
    print(by_target)

    if len(bins) > 0:
        print("\nBy YOLO error bin:")
        print(bins)

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
