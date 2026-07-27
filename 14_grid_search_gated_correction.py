import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common_correction import get_anchor_conf, point_error, compute_image_scale


def choose_point(row, target_conf_thr, anchor_conf_thr, max_shift_ratio, prior_margin_px):
    """
    Conservative gated correction.

    Rule:
    - If YOLO target confidence is high, keep YOLO.
    - If target confidence is low AND anchors reliable AND prior not too far from YOLO, use prior.
    - Otherwise keep YOLO.

    Extra safeguard:
    - If prior is only slightly better in training but moves too far, avoid it.
    """

    target_conf = float(row.get("yolo_target_conf", 0.0))
    anchor_min, _, _ = get_anchor_conf(row, row["target_name"])

    yx = float(row.get("yolo_target_x", np.nan))
    yy = float(row.get("yolo_target_y", np.nan))
    px = float(row.get("prior_pred_x", np.nan))
    py = float(row.get("prior_pred_y", np.nan))

    if np.isnan(yx) or np.isnan(yy):
        return px, py, "prior_no_yolo"

    if np.isnan(px) or np.isnan(py):
        return yx, yy, "yolo_no_prior"

    scale = compute_image_scale(row)
    shift = np.sqrt((px - yx) ** 2 + (py - yy) ** 2)
    shift_ratio = shift / max(scale, 1.0)

    if target_conf < target_conf_thr and anchor_min >= anchor_conf_thr and shift_ratio <= max_shift_ratio:
        return px, py, "prior_gate"

    return yx, yy, "yolo_gate"


def apply_gate(df, target_conf_thr, anchor_conf_thr, max_shift_ratio, prior_margin_px):
    rows = []

    for _, row in df.iterrows():
        out = row.to_dict()

        fx, fy, decision = choose_point(
            row,
            target_conf_thr=target_conf_thr,
            anchor_conf_thr=anchor_conf_thr,
            max_shift_ratio=max_shift_ratio,
            prior_margin_px=prior_margin_px,
        )

        out["final_x"] = fx
        out["final_y"] = fy
        out["gate_decision"] = decision
        out["final_error_px"] = point_error(fx, fy, row["target_orig_x"], row["target_orig_y"])

        rows.append(out)

    return pd.DataFrame(rows)


def split_by_image(df, val_ratio=0.30, seed=42):
    rng = np.random.default_rng(seed)

    if "image_name" not in df.columns:
        mask = rng.random(len(df)) < val_ratio
        return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)

    names = df["image_name"].dropna().unique()
    rng.shuffle(names)

    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])

    val = df[df["image_name"].isin(val_names)].reset_index(drop=True)
    train = df[~df["image_name"].isin(val_names)].reset_index(drop=True)

    return train, val


def summarize(df):
    return {
        "count": len(df),
        "yolo_mean_px": df["yolo_target_error_px"].mean(),
        "prior_mean_px": df["prior_error_px"].mean(),
        "geometry_mean_px": df["geometry_error_px"].mean(),
        "final_mean_px": df["final_error_px"].mean(),
        "yolo_median_px": df["yolo_target_error_px"].median(),
        "prior_median_px": df["prior_error_px"].median(),
        "geometry_median_px": df["geometry_error_px"].median(),
        "final_median_px": df["final_error_px"].median(),
        "final_better_than_yolo": (df["final_error_px"] < df["yolo_target_error_px"]).mean(),
        "prior_use_rate": (df["gate_decision"] == "prior_gate").mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage2c_gated_correction")
    parser.add_argument("--val_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # keep only rows where YOLO person exists
    if "yolo_found_person" in df.columns:
        df = df[df["yolo_found_person"] == 1].reset_index(drop=True)

    df = df.dropna(subset=["yolo_target_error_px", "prior_error_px", "target_orig_x", "target_orig_y"]).reset_index(drop=True)

    train_df, val_df = split_by_image(df, val_ratio=args.val_ratio, seed=args.seed)

    print(f"Total valid rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")

    target_conf_grid = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    anchor_conf_grid = [0.15, 0.25, 0.35, 0.45, 0.55]
    max_shift_grid = [0.10, 0.20, 0.30, 0.45, 0.60, 1.00]
    prior_margin_grid = [0.0]  # reserved for future

    search_rows = []
    best = None
    best_score = float("inf")

    for tc in tqdm(target_conf_grid, desc="Grid search target conf"):
        for ac in anchor_conf_grid:
            for ms in max_shift_grid:
                for pm in prior_margin_grid:
                    train_res = apply_gate(train_df, tc, ac, ms, pm)
                    s = summarize(train_res)
                    score = s["final_mean_px"]

                    row = {
                        "target_conf_thr": tc,
                        "anchor_conf_thr": ac,
                        "max_shift_ratio": ms,
                        "prior_margin_px": pm,
                        **s,
                    }
                    search_rows.append(row)

                    if score < best_score:
                        best_score = score
                        best = row

    search_df = pd.DataFrame(search_rows).sort_values("final_mean_px")
    search_df.to_csv(out_dir / "gated_threshold_search.csv", index=False)

    print("\nBest thresholds on train:")
    print(best)

    val_res = apply_gate(
        val_df,
        target_conf_thr=best["target_conf_thr"],
        anchor_conf_thr=best["anchor_conf_thr"],
        max_shift_ratio=best["max_shift_ratio"],
        prior_margin_px=best["prior_margin_px"],
    )

    all_res = apply_gate(
        df,
        target_conf_thr=best["target_conf_thr"],
        anchor_conf_thr=best["anchor_conf_thr"],
        max_shift_ratio=best["max_shift_ratio"],
        prior_margin_px=best["prior_margin_px"],
    )

    val_summary = summarize(val_res)
    all_summary = summarize(all_res)

    summary_rows = [
        {"split": "train_best_thresholds", **best},
        {"split": "val", **val_summary, 
         "target_conf_thr": best["target_conf_thr"], "anchor_conf_thr": best["anchor_conf_thr"], 
         "max_shift_ratio": best["max_shift_ratio"], "prior_margin_px": best["prior_margin_px"]},
        {"split": "all", **all_summary, 
         "target_conf_thr": best["target_conf_thr"], "anchor_conf_thr": best["anchor_conf_thr"], 
         "max_shift_ratio": best["max_shift_ratio"], "prior_margin_px": best["prior_margin_px"]},
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "gated_summary.csv", index=False)

    all_res.to_csv(out_dir / "best_gated_results.csv", index=False)

    per_target = all_res.groupby("target_name").agg(
        count=("final_error_px", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        prior_mean_px=("prior_error_px", "mean"),
        final_mean_px=("final_error_px", "mean"),
        yolo_median_px=("yolo_target_error_px", "median"),
        prior_median_px=("prior_error_px", "median"),
        final_median_px=("final_error_px", "median"),
        prior_use_rate=("gate_decision", lambda x: (x == "prior_gate").mean()),
        final_better_than_yolo=("final_error_px", lambda x: np.nan),
    ).reset_index()

    # Compute better than yolo per target separately
    better = all_res.groupby("target_name").apply(
        lambda g: (g["final_error_px"] < g["yolo_target_error_px"]).mean()
    ).reset_index(name="final_better_than_yolo")
    per_target = per_target.drop(columns=["final_better_than_yolo"]).merge(better, on="target_name")
    per_target.to_csv(out_dir / "gated_summary_by_target.csv", index=False)

    print("\nValidation summary:")
    print(val_summary)

    print("\nAll-data summary:")
    print(all_summary)

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
