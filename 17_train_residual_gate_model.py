import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.residual_gate_model import ResidualGateMLP
from src.residual_gate_utils import (
    build_feature_and_targets,
    denormalize_xy,
    point_error,
    save_json,
    load_json,
)


class ResidualGateDataset(Dataset):
    def __init__(self, df, bad_error_px=50.0):
        self.df = df.reset_index(drop=True)
        self.bad_error_px = bad_error_px

        feats = []
        residuals = []
        gates = []
        metas = []

        for _, row in self.df.iterrows():
            f, r, g, m = build_feature_and_targets(row, bad_error_px=bad_error_px)
            feats.append(f)
            residuals.append(r)
            gates.append(g)
            metas.append(m)

        self.x = np.stack(feats).astype(np.float32)
        self.residual = np.stack(residuals).astype(np.float32)
        self.gate = np.array(gates, dtype=np.float32)
        self.metas = metas

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.residual[idx], dtype=torch.float32),
            torch.tensor(self.gate[idx], dtype=torch.float32),
            idx,
        )


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


def train_one_epoch(model, loader, optimizer, device, pos_weight, residual_bad_weight):
    model.train()

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    smooth = torch.nn.SmoothL1Loss(reduction="none")

    total = 0.0
    total_res = 0.0
    total_gate = 0.0

    for x, res_target, gate_target, _idx in tqdm(loader, leave=False):
        x = x.to(device)
        res_target = res_target.to(device)
        gate_target = gate_target.to(device)

        pred_res, gate_logit = model(x)

        # gate/classification loss
        gate_loss = bce(gate_logit, gate_target)

        # residual loss: higher weight for bad YOLO cases
        res_loss_raw = smooth(pred_res, res_target).mean(dim=1)
        weights = 1.0 + residual_bad_weight * gate_target
        res_loss = (res_loss_raw * weights).mean()

        loss = res_loss + gate_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)
        total_res += res_loss.item() * x.size(0)
        total_gate += gate_loss.item() * x.size(0)

    return {
        "loss": total / len(loader.dataset),
        "res_loss": total_res / len(loader.dataset),
        "gate_loss": total_gate / len(loader.dataset),
    }


def eval_epoch(model, loader, device, pos_weight, residual_bad_weight):
    model.eval()

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    smooth = torch.nn.SmoothL1Loss(reduction="none")

    total = 0.0
    total_res = 0.0
    total_gate = 0.0
    gate_correct = []

    with torch.no_grad():
        for x, res_target, gate_target, _idx in tqdm(loader, leave=False):
            x = x.to(device)
            res_target = res_target.to(device)
            gate_target = gate_target.to(device)

            pred_res, gate_logit = model(x)

            gate_loss = bce(gate_logit, gate_target)
            res_loss_raw = smooth(pred_res, res_target).mean(dim=1)
            weights = 1.0 + residual_bad_weight * gate_target
            res_loss = (res_loss_raw * weights).mean()

            loss = res_loss + gate_loss

            prob = torch.sigmoid(gate_logit)
            pred_gate = (prob >= 0.5).float()
            gate_correct.append((pred_gate == gate_target).float().cpu())

            total += loss.item() * x.size(0)
            total_res += res_loss.item() * x.size(0)
            total_gate += gate_loss.item() * x.size(0)

    gate_acc = torch.cat(gate_correct).mean().item() if gate_correct else 0.0

    return {
        "loss": total / len(loader.dataset),
        "res_loss": total_res / len(loader.dataset),
        "gate_loss": total_gate / len(loader.dataset),
        "gate_acc": gate_acc,
    }


def raw_predict(model, df, device, bad_error_px):
    ds = ResidualGateDataset(df, bad_error_px=bad_error_px)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    pred_res = np.zeros((len(df), 2), dtype=np.float32)
    pred_gate_prob = np.zeros(len(df), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for x, _res_target, _gate_target, idx in loader:
            x = x.to(device)
            res, logit = model(x)
            prob = torch.sigmoid(logit)

            pred_res[idx.numpy()] = res.cpu().numpy()
            pred_gate_prob[idx.numpy()] = prob.cpu().numpy()

    out = df.copy()
    out["pred_res_dx_norm"] = pred_res[:, 0]
    out["pred_res_dy_norm"] = pred_res[:, 1]
    out["pred_gate_prob"] = pred_gate_prob

    candidate_x = []
    candidate_y = []

    for i, row in out.iterrows():
        _, _, _, meta = build_feature_and_targets(row, bad_error_px=bad_error_px)
        cand_nx = meta["yolo_norm_x"] + row["pred_res_dx_norm"]
        cand_ny = meta["yolo_norm_y"] + row["pred_res_dy_norm"]
        px, py = denormalize_xy(cand_nx, cand_ny, meta["width"], meta["height"])
        candidate_x.append(px)
        candidate_y.append(py)

    out["candidate_x"] = candidate_x
    out["candidate_y"] = candidate_y
    out["candidate_error_px"] = np.sqrt(
        (out["candidate_x"] - out["target_orig_x"]) ** 2
        + (out["candidate_y"] - out["target_orig_y"]) ** 2
    )

    return out


def apply_threshold(df, gate_thr=0.70, max_shift_ratio=0.45, blend=True, bad_error_px=50.0):
    rows = []

    for _, row in df.iterrows():
        out = row.to_dict()

        yx = row["yolo_target_x"]
        yy = row["yolo_target_y"]
        cx = row["candidate_x"]
        cy = row["candidate_y"]

        _, _, _, meta = build_feature_and_targets(row, bad_error_px=bad_error_px)
        scale = meta["scale"]

        shift = np.sqrt((cx - yx) ** 2 + (cy - yy) ** 2)
        shift_ratio = shift / max(scale, 1.0)

        use_corr = (row["pred_gate_prob"] >= gate_thr) and (shift_ratio <= max_shift_ratio)

        if use_corr:
            if blend:
                alpha = float(row["pred_gate_prob"])
                fx = yx + alpha * (cx - yx)
                fy = yy + alpha * (cy - yy)
                decision = "residual_blend"
            else:
                fx, fy = cx, cy
                decision = "residual_hard"
        else:
            fx, fy = yx, yy
            decision = "keep_yolo"

        out["final_x"] = fx
        out["final_y"] = fy
        out["final_error_px"] = point_error(fx, fy, row["target_orig_x"], row["target_orig_y"])
        out["final_decision"] = decision
        out["final_shift_ratio"] = shift_ratio

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df["final_better_than_yolo"] = out_df["final_error_px"] < out_df["yolo_target_error_px"]
    out_df["candidate_better_than_yolo"] = out_df["candidate_error_px"] < out_df["yolo_target_error_px"]

    return out_df


def summarize(df):
    return {
        "count": len(df),
        "yolo_mean_px": df["yolo_target_error_px"].mean(),
        "candidate_mean_px": df["candidate_error_px"].mean(),
        "prior_mean_px": df["prior_error_px"].mean(),
        "geometry_mean_px": df["geometry_error_px"].mean(),
        "final_mean_px": df["final_error_px"].mean(),
        "yolo_median_px": df["yolo_target_error_px"].median(),
        "candidate_median_px": df["candidate_error_px"].median(),
        "final_median_px": df["final_error_px"].median(),
        "candidate_win_vs_yolo": df["candidate_better_than_yolo"].mean(),
        "final_win_vs_yolo": df["final_better_than_yolo"].mean(),
        "correction_use_rate": (df["final_decision"] != "keep_yolo").mean(),
    }


def grid_search_threshold(train_raw, val_raw, bad_error_px):
    gate_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    shift_grid = [0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00]
    blend_grid = [True, False]

    rows = []
    best = None
    best_score = float("inf")

    for gate_thr in gate_grid:
        for max_shift in shift_grid:
            for blend in blend_grid:
                res = apply_threshold(
                    train_raw,
                    gate_thr=gate_thr,
                    max_shift_ratio=max_shift,
                    blend=blend,
                    bad_error_px=bad_error_px,
                )
                s = summarize(res)
                row = {
                    "gate_thr": gate_thr,
                    "max_shift_ratio": max_shift,
                    "blend": blend,
                    **s,
                }
                rows.append(row)

                # choose by train final mean, but penalize if it changes too many points
                score = s["final_mean_px"] + 2.0 * max(0.0, s["correction_use_rate"] - 0.50)
                if score < best_score:
                    best_score = score
                    best = row

    search = pd.DataFrame(rows).sort_values("final_mean_px")

    val_res = apply_threshold(
        val_raw,
        gate_thr=best["gate_thr"],
        max_shift_ratio=best["max_shift_ratio"],
        blend=bool(best["blend"]),
        bad_error_px=bad_error_px,
    )

    return search, best, val_res


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage2d_residual_gate")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--bad_error_px", type=float, default=50.0)
    parser.add_argument("--residual_bad_weight", type=float, default=3.0)
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

    required = [
        "target_orig_x", "target_orig_y",
        "yolo_target_x", "yolo_target_y", "yolo_target_error_px",
        "prior_pred_x", "prior_pred_y", "prior_error_px",
        "geometry_x", "geometry_y", "geometry_error_px",
    ]
    df = df.dropna(subset=required).reset_index(drop=True)

    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = ResidualGateDataset(train_df, bad_error_px=args.bad_error_px)
    val_ds = ResidualGateDataset(val_df, bad_error_px=args.bad_error_px)

    input_dim = train_ds.x.shape[1]

    bad_rate = float(train_ds.gate.mean())
    pos = max(train_ds.gate.sum(), 1.0)
    neg = max(len(train_ds.gate) - train_ds.gate.sum(), 1.0)
    pos_weight_value = neg / pos

    print(f"Total valid samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Input dim: {input_dim}")
    print(f"Bad YOLO threshold: {args.bad_error_px}px")
    print(f"Bad rate in train: {bad_rate:.3f}")
    print(f"Gate pos_weight: {pos_weight_value:.3f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResidualGateMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, optimizer, device,
            pos_weight=pos_weight,
            residual_bad_weight=args.residual_bad_weight,
        )
        va = eval_epoch(
            model, val_loader, device,
            pos_weight=pos_weight,
            residual_bad_weight=args.residual_bad_weight,
        )

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_res_loss": tr["res_loss"],
            "train_gate_loss": tr["gate_loss"],
            "val_loss": va["loss"],
            "val_res_loss": va["res_loss"],
            "val_gate_loss": va["gate_loss"],
            "val_gate_acc": va["gate_acc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train={tr['loss']:.5f} | "
            f"val={va['loss']:.5f} | "
            f"val_gate_acc={va['gate_acc']:.3f}"
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            save_json({
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "bad_error_px": args.bad_error_px,
                "residual_bad_weight": args.residual_bad_weight,
                "best_val_loss": best_val,
            }, out_dir / "model_config.json")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    # Load best model
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    train_raw = raw_predict(model, train_df, device, bad_error_px=args.bad_error_px)
    val_raw = raw_predict(model, val_df, device, bad_error_px=args.bad_error_px)
    all_raw = raw_predict(model, df, device, bad_error_px=args.bad_error_px)

    search, best_thr, val_final = grid_search_threshold(train_raw, val_raw, bad_error_px=args.bad_error_px)
    search.to_csv(out_dir / "residual_gate_threshold_search.csv", index=False)

    train_final = apply_threshold(
        train_raw,
        gate_thr=best_thr["gate_thr"],
        max_shift_ratio=best_thr["max_shift_ratio"],
        blend=bool(best_thr["blend"]),
        bad_error_px=args.bad_error_px,
    )
    all_final = apply_threshold(
        all_raw,
        gate_thr=best_thr["gate_thr"],
        max_shift_ratio=best_thr["max_shift_ratio"],
        blend=bool(best_thr["blend"]),
        bad_error_px=args.bad_error_px,
    )

    train_final.to_csv(out_dir / "train_predictions.csv", index=False)
    val_final.to_csv(out_dir / "val_predictions.csv", index=False)
    all_final.to_csv(out_dir / "all_predictions.csv", index=False)

    summary = pd.DataFrame([
        {"split": "train", "gate_thr": best_thr["gate_thr"], "max_shift_ratio": best_thr["max_shift_ratio"], "blend": bool(best_thr["blend"]), **summarize(train_final)},
        {"split": "val", "gate_thr": best_thr["gate_thr"], "max_shift_ratio": best_thr["max_shift_ratio"], "blend": bool(best_thr["blend"]), **summarize(val_final)},
        {"split": "all", "gate_thr": best_thr["gate_thr"], "max_shift_ratio": best_thr["max_shift_ratio"], "blend": bool(best_thr["blend"]), **summarize(all_final)},
    ])
    summary.to_csv(out_dir / "residual_gate_summary.csv", index=False)

    per_target = all_final.groupby("target_name").agg(
        count=("final_error_px", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        candidate_mean_px=("candidate_error_px", "mean"),
        final_mean_px=("final_error_px", "mean"),
        yolo_median_px=("yolo_target_error_px", "median"),
        final_median_px=("final_error_px", "median"),
        candidate_win_vs_yolo=("candidate_better_than_yolo", "mean"),
        final_win_vs_yolo=("final_better_than_yolo", "mean"),
        correction_use_rate=("final_decision", lambda x: (x != "keep_yolo").mean()),
    ).reset_index()
    per_target.to_csv(out_dir / "residual_gate_summary_by_target.csv", index=False)

    print("\nBest threshold from train:")
    print(best_thr)

    print("\nSummary:")
    print(summary)

    print("\nBy target:")
    print(per_target)

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
