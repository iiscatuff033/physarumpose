import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.multihypothesis_utils import (
    select_diverse_topk,
    softmax,
    mean_pairwise_distance,
    min_error_topk,
    hit_topk,
    add_yolo_bins,
    group_summary,
)


def make_topk_tables(cand_df, top_k=5, nms_dist=12.0, temperature=0.10):
    long_rows = []
    wide_rows = []

    for sample_id, g in tqdm(cand_df.groupby("sample_id"), desc="Selecting top-K hypotheses"):
        selected = select_diverse_topk(
            group=g,
            top_k=top_k,
            nms_dist=nms_dist,
            score_col="learned_score",
        )

        if len(selected) == 0:
            continue

        probs = softmax(selected["learned_score"].values, temperature=temperature)

        # Long rows: one row per selected hypothesis.
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            out = row.to_dict()
            out["rank"] = rank
            out["hypothesis_probability"] = float(probs[rank - 1])
            out["top_k"] = int(top_k)
            out["nms_dist"] = float(nms_dist)
            out["temperature"] = float(temperature)
            long_rows.append(out)

        first = selected.iloc[0]

        # Diversity numbers
        pts_top3 = selected.head(min(3, len(selected)))[["candidate_x", "candidate_y"]].values
        pts_top5 = selected.head(min(5, len(selected)))[["candidate_x", "candidate_y"]].values

        wide = {
            "sample_id": sample_id,
            "image_name": first.get("image_name", ""),
            "occluded_image_name": first.get("occluded_image_name", ""),
            "target_name": first["target_name"],

            "top1_x": float(first["candidate_x"]),
            "top1_y": float(first["candidate_y"]),
            "top1_error_px": float(first["candidate_error_px"]),
            "top1_score": float(first["learned_score"]),
            "top1_probability": float(probs[0]),

            "top3_oracle_error_px": min_error_topk(selected, 3),
            "top5_oracle_error_px": min_error_topk(selected, 5),
            "top3_hit_10px": hit_topk(selected, 3, 10),
            "top3_hit_20px": hit_topk(selected, 3, 20),
            "top3_hit_30px": hit_topk(selected, 3, 30),
            "top5_hit_10px": hit_topk(selected, 5, 10),
            "top5_hit_20px": hit_topk(selected, 5, 20),
            "top5_hit_30px": hit_topk(selected, 5, 30),

            "top3_diversity_px": mean_pairwise_distance(pts_top3),
            "top5_diversity_px": mean_pairwise_distance(pts_top5),

            "target_orig_x": float(first["target_orig_x"]),
            "target_orig_y": float(first["target_orig_y"]),
            "anchor1_x": float(first["anchor1_x"]),
            "anchor1_y": float(first["anchor1_y"]),
            "anchor2_x": float(first["anchor2_x"]),
            "anchor2_y": float(first["anchor2_y"]),
            "anchor1_conf": float(first.get("anchor1_conf", np.nan)),
            "anchor2_conf": float(first.get("anchor2_conf", np.nan)),
            "anchor_min_conf": float(first.get("anchor_min_conf", np.nan)),
        }

        # Add top-K coordinate/error columns in wide format.
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            wide[f"top{rank}_x"] = float(row["candidate_x"])
            wide[f"top{rank}_y"] = float(row["candidate_y"])
            wide[f"top{rank}_error_px"] = float(row["candidate_error_px"])
            wide[f"top{rank}_score"] = float(row["learned_score"])
            wide[f"top{rank}_probability"] = float(probs[rank - 1])

        # Baseline columns if available.
        for col in [
            "yolo_target_x",
            "yolo_target_y",
            "yolo_target_error_px",
            "region_pred_x",
            "region_pred_y",
            "region_pred_conf",
            "region_error_px",
            "coarse_region_x",
            "coarse_region_y",
            "coarse_region_conf",
            "coarse_cell_px",
            "coarse_region_error_px",
            "nearest_coarse_x",
            "nearest_coarse_y",
            "nearest_coarse_error_px",
            "heuristic_x",
            "heuristic_y",
            "heuristic_error_px",
            "oracle_x",
            "oracle_y",
            "oracle_error_px",
        ]:
            if col in first.index:
                wide[col] = first[col]

        if "yolo_target_error_px" in wide:
            wide["top1_better_than_yolo"] = wide["top1_error_px"] < wide["yolo_target_error_px"]

        wide_rows.append(wide)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)

    return long_df, wide_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5e_multihypothesis_slime")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--nms_dist", type=float, default=12.0)
    parser.add_argument("--temperature", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_df = pd.read_csv(args.candidate_csv)

    required = ["sample_id", "candidate_x", "candidate_y", "candidate_error_px", "learned_score"]
    missing = [c for c in required if c not in cand_df.columns]
    if missing:
        raise ValueError(f"Candidate CSV is missing required columns: {missing}")

    long_df, wide_df = make_topk_tables(
        cand_df=cand_df,
        top_k=args.top_k,
        nms_dist=args.nms_dist,
        temperature=args.temperature,
    )

    long_df.to_csv(out_dir / "topk_predictions_long.csv", index=False)
    wide_df.to_csv(out_dir / "topk_predictions_wide.csv", index=False)

    summary = pd.DataFrame([{
        "split": "all",
        "top_k": args.top_k,
        "nms_dist": args.nms_dist,
        "temperature": args.temperature,
        **group_summary(wide_df),
    }])
    summary.to_csv(out_dir / "multihypothesis_summary.csv", index=False)

    by_target = []
    for target_name, g in wide_df.groupby("target_name"):
        row = {
            "target_name": target_name,
            "top_k": args.top_k,
            "nms_dist": args.nms_dist,
            "temperature": args.temperature,
            **group_summary(g),
        }
        by_target.append(row)
    by_target = pd.DataFrame(by_target)
    by_target.to_csv(out_dir / "multihypothesis_summary_by_target.csv", index=False)

    bins = add_yolo_bins(wide_df)
    if len(bins) > 0:
        bins.to_csv(out_dir / "multihypothesis_error_bins.csv", index=False)

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
