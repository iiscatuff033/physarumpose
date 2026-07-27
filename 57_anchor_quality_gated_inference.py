import argparse
from pathlib import Path
import itertools

import pandas as pd
import numpy as np

from src.gate_utils import (
    save_json,
    split_by_image,
    merge_pred_and_topk,
    apply_gate,
    summarize,
    error_bins,
)


def grid_search_gate(df):
    """
    Tune a simple interpretable gate.

    The gate does not use GT at inference. GT is only used here to choose
    thresholds on a calibration/tune split.
    """

    min_anchor_dist_list = [10, 15, 20, 25, 30, 40]
    max_anchor_dist_list = [80, 120, 160, 220, 300, 99999]
    min_anchor_conf_list = [0.00, 0.05, 0.10, 0.20, 0.35]
    min_top1_prob_list = [0.00, 0.05, 0.10, 0.15, 0.20]
    min_prob_margin_list = [-1.0, 0.00, 0.02, 0.05, 0.10]
    max_top1_region_dist_list = [15, 25, 35, 50, 70, 100, 99999]
    max_top1_soft_dist_list = [15, 25, 35, 50, 70, 100, 99999]

    rows = []
    best = None

    combos = itertools.product(
        min_anchor_dist_list,
        max_anchor_dist_list,
        min_anchor_conf_list,
        min_top1_prob_list,
        min_prob_margin_list,
        max_top1_region_dist_list,
        max_top1_soft_dist_list,
    )

    for (
        min_anchor_dist,
        max_anchor_dist,
        min_anchor_conf,
        min_top1_prob,
        min_prob_margin,
        max_top1_region_dist,
        max_top1_soft_dist,
    ) in combos:

        if max_anchor_dist <= min_anchor_dist:
            continue

        cfg = {
            "min_anchor_dist": float(min_anchor_dist),
            "max_anchor_dist": float(max_anchor_dist),
            "min_anchor_conf": float(min_anchor_conf),
            "min_top1_prob": float(min_top1_prob),
            "min_prob_margin": float(min_prob_margin),
            "max_top1_region_dist": float(max_top1_region_dist),
            "max_top1_soft_dist": float(max_top1_soft_dist),
        }

        g = apply_gate(df, cfg)
        s = summarize(g)

        row = dict(cfg)
        row.update({
            "gated_mean_px": s["gated_mean_px"],
            "gated_median_px": s["gated_median_px"],
            "gate_use_top1_rate": s["gate_use_top1_rate"],
            "gated_win_vs_region": s["gated_win_vs_region"],
            "gated_win_vs_top1": s["gated_win_vs_top1"],
            "objective": s["gated_mean_px"],
        })

        if "gated_win_vs_yolo" in s:
            row["gated_win_vs_yolo"] = s["gated_win_vs_yolo"]

        rows.append(row)

        if best is None or row["objective"] < best["objective"]:
            best = row

    search_df = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    best_cfg = {
        "min_anchor_dist": float(best["min_anchor_dist"]),
        "max_anchor_dist": float(best["max_anchor_dist"]),
        "min_anchor_conf": float(best["min_anchor_conf"]),
        "min_top1_prob": float(best["min_top1_prob"]),
        "min_prob_margin": float(best["min_prob_margin"]),
        "max_top1_region_dist": float(best["max_top1_region_dist"]),
        "max_top1_soft_dist": float(best["max_top1_soft_dist"]),
    }

    return search_df, best_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--topk_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5fc_anchor_quality_gate")
    parser.add_argument("--tune_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_simple_top1", action="store_true",
                        help="Skip grid search and always use top1 path.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(args.pred_csv)
    topk_df = pd.read_csv(args.topk_csv)

    merged = merge_pred_and_topk(pred_df, topk_df)
    merged.to_csv(out_dir / "gate_input_merged.csv", index=False)

    tune_df, holdout_df = split_by_image(merged, tune_ratio=args.tune_ratio, seed=args.seed)

    if args.use_simple_top1:
        best_cfg = {
            "min_anchor_dist": 0.0,
            "max_anchor_dist": 999999.0,
            "min_anchor_conf": 0.0,
            "min_top1_prob": 0.0,
            "min_prob_margin": -1.0,
            "max_top1_region_dist": 999999.0,
            "max_top1_soft_dist": 999999.0,
        }
        search_df = pd.DataFrame([best_cfg])
        search_df["objective"] = np.nan
    else:
        search_df, best_cfg = grid_search_gate(tune_df)

    search_df.to_csv(out_dir / "gate_search_results.csv", index=False)
    save_json(best_cfg, out_dir / "best_gate_config.json")

    all_gated = apply_gate(merged, best_cfg)
    tune_gated = apply_gate(tune_df, best_cfg)
    holdout_gated = apply_gate(holdout_df, best_cfg)

    all_gated.to_csv(out_dir / "gated_predictions.csv", index=False)
    tune_gated.to_csv(out_dir / "gated_predictions_tune.csv", index=False)
    holdout_gated.to_csv(out_dir / "gated_predictions_holdout.csv", index=False)

    summary = pd.DataFrame([
        {"split": "tune", **summarize(tune_gated)},
        {"split": "holdout", **summarize(holdout_gated)},
        {"split": "all", **summarize(all_gated)},
    ])
    summary.to_csv(out_dir / "gated_summary.csv", index=False)

    by_target = []
    for target_name, g in all_gated.groupby("target_name"):
        row = {"target_name": target_name}
        row.update(summarize(g))
        by_target.append(row)
    by_target = pd.DataFrame(by_target)
    by_target.to_csv(out_dir / "gated_summary_by_target.csv", index=False)

    bins = error_bins(all_gated)
    if len(bins) > 0:
        bins.to_csv(out_dir / "gated_error_bins.csv", index=False)

    print("\nBest gate config:")
    print(best_cfg)

    print("\nSummary:")
    print(summary.T)

    print("\nBy target:")
    print(by_target)

    if len(bins) > 0:
        print("\nBy YOLO error bin:")
        print(bins)

    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    main()
