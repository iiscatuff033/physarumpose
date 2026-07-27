import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.learned_slime_model import LearnedSlimeScorer
from src.learned_slime_utils import load_json, save_json, summarize_predictions


class CandidateDataset(Dataset):
    def __init__(self, df, feature_names):
        self.df = df.reset_index(drop=True)
        self.feature_names = feature_names
        x = self.df[feature_names].values.astype(np.float32)
        y = self.df["label_quality"].values.astype(np.float32)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), idx


def split_by_image(candidate_df, val_ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)
    base = candidate_df[["sample_id", "image_name"]].drop_duplicates("sample_id")
    names = base["image_name"].fillna("").unique()
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])

    val_samples = set(base[base["image_name"].isin(val_names)]["sample_id"].values.tolist())
    val_df = candidate_df[candidate_df["sample_id"].isin(val_samples)].reset_index(drop=True)
    train_df = candidate_df[~candidate_df["sample_id"].isin(val_samples)].reset_index(drop=True)
    return train_df, val_df


def train_epoch(model, loader, optimizer, device, good_weight=3.0):
    model.train()
    total = 0.0
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    for x, y, _idx in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss_raw = bce(pred, y)
        weights = 1.0 + good_weight * y
        loss = (loss_raw * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def eval_epoch(model, loader, device, good_weight=3.0):
    model.eval()
    total = 0.0
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for x, y, _idx in tqdm(loader, leave=False):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss_raw = bce(pred, y)
            weights = 1.0 + good_weight * y
            loss = (loss_raw * weights).mean()
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def predict_candidates(model, df, feature_names, device, batch_size=8192):
    ds = CandidateDataset(df, feature_names)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    scores = np.zeros(len(df), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for x, _y, idx in tqdm(loader, leave=False, desc="Predicting candidates"):
            x = x.to(device)
            logit = model(x)
            prob = torch.sigmoid(logit).cpu().numpy()
            scores[idx.numpy()] = prob

    out = df.copy()
    out["learned_score"] = scores
    return out


def select_best_per_sample(candidate_pred_df):
    rows = []
    for sample_id, g in candidate_pred_df.groupby("sample_id"):
        best = g.sort_values("learned_score", ascending=False).iloc[0].to_dict()
        heur = g.sort_values("slime_reinforced_score", ascending=False).iloc[0].to_dict()

        out = best.copy()
        out["learned_pred_x"] = best["candidate_x"]
        out["learned_pred_y"] = best["candidate_y"]
        out["learned_error_px"] = best["candidate_error_px"]
        out["heuristic_pred_x"] = heur["candidate_x"]
        out["heuristic_pred_y"] = heur["candidate_y"]
        out["heuristic_error_px"] = heur["candidate_error_px"]
        out["heuristic_score"] = heur["slime_reinforced_score"]
        out["oracle_best_error_px"] = g["candidate_error_px"].min()
        rows.append(out)

    result = pd.DataFrame(rows)
    if "yolo_target_error_px" in result.columns:
        result["learned_better_than_yolo"] = result["learned_error_px"] < result["yolo_target_error_px"]
        result["heuristic_better_than_yolo"] = result["heuristic_error_px"] < result["yolo_target_error_px"]
    result["learned_better_than_heuristic"] = result["learned_error_px"] < result["heuristic_error_px"]
    return result


def make_error_bins(best_df):
    if "yolo_target_error_px" not in best_df.columns:
        return pd.DataFrame()
    bins = [0, 20, 40, 60, 80, 100, 150, 999999]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"]
    df = best_df.copy()
    df["yolo_error_bin"] = pd.cut(df["yolo_target_error_px"], bins=bins, labels=labels, include_lowest=True, right=False)
    out = df.groupby("yolo_error_bin", observed=False).agg(
        count=("learned_error_px", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        heuristic_mean_px=("heuristic_error_px", "mean"),
        learned_mean_px=("learned_error_px", "mean"),
        oracle_mean_px=("oracle_best_error_px", "mean"),
        yolo_median_px=("yolo_target_error_px", "median"),
        heuristic_median_px=("heuristic_error_px", "median"),
        learned_median_px=("learned_error_px", "median"),
        learned_win_vs_yolo=("learned_better_than_yolo", "mean"),
        learned_win_vs_heuristic=("learned_better_than_heuristic", "mean"),
    ).reset_index()
    return out


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_csv", type=str, required=True)
    parser.add_argument("--feature_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5_learned_slime_scorer")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--good_weight", type=float, default=3.0)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = pd.read_csv(args.candidate_csv)
    feature_names = load_json(args.feature_json)["feature_names"]

    # clean feature NaNs
    candidate_df[feature_names] = candidate_df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    candidate_df = candidate_df.dropna(subset=["label_quality", "candidate_error_px"]).reset_index(drop=True)

    train_df, val_df = split_by_image(candidate_df, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = CandidateDataset(train_df, feature_names)
    val_ds = CandidateDataset(val_df, feature_names)

    print(f"Candidate rows: {len(candidate_df)}")
    print(f"Train rows: {len(train_df)} from {train_df['sample_id'].nunique()} samples")
    print(f"Val rows: {len(val_df)} from {val_df['sample_id'].nunique()} samples")
    print(f"Input dim: {len(feature_names)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LearnedSlimeScorer(input_dim=len(feature_names), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device, good_weight=args.good_weight)
        va_loss = eval_epoch(model, val_loader, device, good_weight=args.good_weight)
        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss}
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            save_json({
                "input_dim": len(feature_names),
                "feature_names": feature_names,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "best_val_loss": best_val,
            }, out_dir / "model_config.json")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    train_df.to_csv(out_dir / "train_candidates.csv", index=False)
    val_df.to_csv(out_dir / "val_candidates.csv", index=False)

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    all_pred = predict_candidates(model, candidate_df, feature_names, device)
    train_pred = all_pred[all_pred["sample_id"].isin(set(train_df["sample_id"]))].reset_index(drop=True)
    val_pred = all_pred[all_pred["sample_id"].isin(set(val_df["sample_id"]))].reset_index(drop=True)

    all_pred.to_csv(out_dir / "all_candidate_predictions.csv", index=False)
    train_pred.to_csv(out_dir / "train_candidate_predictions.csv", index=False)
    val_pred.to_csv(out_dir / "val_candidate_predictions.csv", index=False)

    train_best = select_best_per_sample(train_pred)
    val_best = select_best_per_sample(val_pred)
    all_best = select_best_per_sample(all_pred)

    train_best.to_csv(out_dir / "train_best_predictions.csv", index=False)
    val_best.to_csv(out_dir / "val_best_predictions.csv", index=False)
    all_best.to_csv(out_dir / "all_best_predictions.csv", index=False)

    summary = pd.DataFrame([
        {"split": "train", **summarize_predictions(train_best)},
        {"split": "val", **summarize_predictions(val_best)},
        {"split": "all", **summarize_predictions(all_best)},
    ])
    summary.to_csv(out_dir / "learned_slime_summary.csv", index=False)

    by_target = all_best.groupby("target_name").agg(
        count=("learned_error_px", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        heuristic_mean_px=("heuristic_error_px", "mean"),
        learned_mean_px=("learned_error_px", "mean"),
        oracle_mean_px=("oracle_best_error_px", "mean"),
        yolo_median_px=("yolo_target_error_px", "median"),
        heuristic_median_px=("heuristic_error_px", "median"),
        learned_median_px=("learned_error_px", "median"),
        learned_win_vs_yolo=("learned_better_than_yolo", "mean"),
        learned_win_vs_heuristic=("learned_better_than_heuristic", "mean"),
    ).reset_index()
    by_target.to_csv(out_dir / "learned_slime_summary_by_target.csv", index=False)

    bins = make_error_bins(all_best)
    bins.to_csv(out_dir / "learned_slime_error_bins.csv", index=False)

    print("\nSummary:")
    print(summary)
    print("\nBy target:")
    print(by_target)
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
