import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.anchor_slime_utils import (
    get_anchor_points,
    is_valid_point,
    generate_candidates,
    score_candidates,
    point_error,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage4_anchor_slime")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--anchor_conf_thresh", type=float, default=0.10)
    parser.add_argument("--edge_w", type=float, default=1.0)
    parser.add_argument("--anatomy_w", type=float, default=1.0)
    parser.add_argument("--occ_w", type=float, default=1.2)
    parser.add_argument("--prior_w", type=float, default=0.25)
    parser.add_argument("--slime_iters", type=int, default=8)
    parser.add_argument("--slime_diffusion", type=float, default=0.20)
    parser.add_argument("--slime_decay", type=float, default=0.08)
    parser.add_argument("--edge_blur", type=int, default=3)
    parser.add_argument("--edge_low", type=int, default=50)
    parser.add_argument("--edge_high", type=int, default=150)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cand_dir = out_dir / "candidate_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    image_dir = Path(args.image_dir)
    weights = vars(args)

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="AnchorNet slime recovery"):
        out = row.to_dict()
        image_path = image_dir / row["occluded_image_name"]
        if not image_path.exists():
            out["anchor_slime_status"] = "missing_image"
            rows.append(out)
            continue
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            out["anchor_slime_status"] = "bad_image"
            rows.append(out)
            continue

        a, b, ac, bc, a_name, b_name = get_anchor_points(row)
        out["anchor1_name"] = a_name
        out["anchor2_name"] = b_name
        out["anchor1_x_used"] = float(a[0]) if np.isfinite(a[0]) else np.nan
        out["anchor1_y_used"] = float(a[1]) if np.isfinite(a[1]) else np.nan
        out["anchor2_x_used"] = float(b[0]) if np.isfinite(b[0]) else np.nan
        out["anchor2_y_used"] = float(b[1]) if np.isfinite(b[1]) else np.nan
        out["anchor1_conf_used"] = float(ac)
        out["anchor2_conf_used"] = float(bc)
        out["anchor_min_conf_used"] = float(min(ac, bc))

        if not is_valid_point(a) or not is_valid_point(b):
            out["anchor_slime_status"] = "bad_anchors"
            rows.append(out)
            continue
        if min(ac, bc) < args.anchor_conf_thresh:
            out["anchor_slime_status"] = "low_anchor_conf"
            rows.append(out)
            continue

        box = None
        if all(k in row for k in ["occ_x1", "occ_y1", "occ_x2", "occ_y2"]):
            box = (row["occ_x1"], row["occ_y1"], row["occ_x2"], row["occ_y2"])
        candidates = generate_candidates(a, b, image_bgr.shape, box=box)
        if len(candidates) == 0:
            out["anchor_slime_status"] = "no_candidates"
            rows.append(out)
            continue

        cand_df = score_candidates(row, image_bgr, candidates, a, b, weights)
        if len(cand_df) == 0:
            out["anchor_slime_status"] = "no_scored_candidates"
            rows.append(out)
            continue
        best = cand_df.sort_values("slime_score", ascending=False).iloc[0]
        sx, sy = float(best["x"]), float(best["y"])

        out["anchor_slime_status"] = "ok"
        out["anchor_slime_pred_x"] = sx
        out["anchor_slime_pred_y"] = sy
        out["anchor_slime_score"] = float(best["slime_score"])
        out["anchor_slime_raw_score"] = float(best["raw_score"])
        out["anchor_slime_edge_score"] = float(best["edge_score"])
        out["anchor_slime_anatomy_score"] = float(best["anatomy_score"])
        out["anchor_slime_occ_score"] = float(best["occ_score"])
        out["anchor_slime_prior_score"] = float(best["prior_score"])
        out["anchor_slime_num_candidates"] = int(len(cand_df))
        out["anchor_slime_error_px"] = point_error(sx, sy, row["target_orig_x"], row["target_orig_y"])

        if "yolo_target_error_px" in row:
            out["anchor_slime_better_than_yolo"] = out["anchor_slime_error_px"] < row["yolo_target_error_px"]
        if "slime_error_px" in row:
            out["anchor_slime_better_than_yolo_anchor_slime"] = out["anchor_slime_error_px"] < row["slime_error_px"]
        if "prior_error_px" in row:
            out["anchor_slime_better_than_prior"] = out["anchor_slime_error_px"] < row["prior_error_px"]
        if "geometry_error_px" in row:
            out["anchor_slime_better_than_geometry"] = out["anchor_slime_error_px"] < row["geometry_error_px"]

        if idx < 80:
            cand_df.sort_values("slime_score", ascending=False).head(80).to_csv(cand_dir / f"anchor_candidates_{idx:05d}_{row['target_name']}.csv", index=False)

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / "anchor_slime_results.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    valid = out_df[out_df["anchor_slime_status"] == "ok"]
    print(f"Valid: {len(valid)} / {len(out_df)}")
    if len(valid) > 0:
        print(valid.groupby("target_name")["anchor_slime_error_px"].agg(["count", "mean", "median"]))


if __name__ == "__main__":
    main()
