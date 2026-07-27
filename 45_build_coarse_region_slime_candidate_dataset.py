import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.coarse_region_utils import (
    get_target_anchors,
    valid_point,
    quantize_pred_region,
    generate_candidates_coarse,
    build_candidate_features,
    COARSE_REGION_FEATURES,
    save_json,
    point_error,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5d_coarse_region_candidates")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--cell_px", type=float, default=64.0)
    parser.add_argument("--jitter_px", type=float, default=0.0)
    parser.add_argument("--anchor_conf_thresh", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    df = pd.read_csv(args.csv_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    all_cands = []
    status_rows = []
    valid_samples = 0

    for sample_id, row in tqdm(df.iterrows(), total=len(df), desc="Building Stage 5D coarse-region candidates"):
        row = row.copy()
        row["sample_id"] = sample_id

        image_path = image_dir / str(row.get("occluded_image_name", ""))
        if not image_path.exists():
            status_rows.append({"sample_id": sample_id, "status": "missing_image"})
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            status_rows.append({"sample_id": sample_id, "status": "bad_image"})
            continue

        try:
            _a_name, _b_name, a, b, ac, bc = get_target_anchors(row)
        except Exception as e:
            status_rows.append({"sample_id": sample_id, "status": f"anchor_parse_error:{e}"})
            continue

        if not valid_point(a) or not valid_point(b):
            status_rows.append({"sample_id": sample_id, "status": "bad_anchor_points"})
            continue

        if ac < args.anchor_conf_thresh or bc < args.anchor_conf_thresh:
            status_rows.append({"sample_id": sample_id, "status": "low_anchor_conf"})
            continue

        coarse_region, coarse_conf = quantize_pred_region(
            row=row,
            image_shape=image_bgr.shape,
            cell_px=args.cell_px,
            jitter_px=args.jitter_px,
            seed=args.seed,
        )

        candidates = generate_candidates_coarse(
            a=a,
            b=b,
            image_shape=image_bgr.shape,
            coarse_region=coarse_region,
            cell_px=args.cell_px,
        )

        if len(candidates) == 0:
            status_rows.append({"sample_id": sample_id, "status": "no_candidates"})
            continue

        cand_df = build_candidate_features(
            row=row,
            image_bgr=image_bgr,
            candidates=candidates,
            a=a,
            b=b,
            ac=ac,
            bc=bc,
            coarse_region=coarse_region,
            coarse_conf=coarse_conf,
            cell_px=args.cell_px,
        )

        if len(cand_df) == 0:
            status_rows.append({"sample_id": sample_id, "status": "empty_candidate_features"})
            continue

        all_cands.append(cand_df)
        valid_samples += 1

        tx = float(row.get("target_orig_x", float("nan")))
        ty = float(row.get("target_orig_y", float("nan")))
        cr_err = point_error(float(coarse_region[0]), float(coarse_region[1]), tx, ty)

        status_rows.append({
            "sample_id": sample_id,
            "status": "ok",
            "num_candidates": len(cand_df),
            "coarse_region_error_px": cr_err,
        })

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(out_dir / "coarse_region_candidate_status.csv", index=False)

    if not all_cands:
        print("No candidate rows were created.")
        print(status_df["status"].value_counts() if "status" in status_df.columns else status_df)
        return

    cand_all = pd.concat(all_cands, ignore_index=True)
    cand_path = out_dir / "coarse_region_candidate_dataset.csv"
    cand_all.to_csv(cand_path, index=False)

    save_json(COARSE_REGION_FEATURES, out_dir / "coarse_region_feature_names.json")

    # Per-sample coarse-region errors
    best_coarse = cand_all.groupby("sample_id").first().reset_index()
    best_coarse["coarse_region_error_px"] = (
        (best_coarse["coarse_region_x"] - best_coarse["target_orig_x"]) ** 2
        + (best_coarse["coarse_region_y"] - best_coarse["target_orig_y"]) ** 2
    ) ** 0.5

    summary = pd.DataFrame([{
        "input_samples": len(df),
        "valid_samples": valid_samples,
        "candidate_rows": len(cand_all),
        "avg_candidates_per_valid_sample": len(cand_all) / max(valid_samples, 1),
        "num_features": len(COARSE_REGION_FEATURES),
        "cell_px": args.cell_px,
        "jitter_px": args.jitter_px,
        "mean_coarse_region_error_px": best_coarse["coarse_region_error_px"].mean(),
        "median_coarse_region_error_px": best_coarse["coarse_region_error_px"].median(),
        "uses_exact_region_point_as_feature": 0,
        "uses_gt_occlusion_box_features": 0,
        "uses_coarse_predicted_region_features": 1,
    }])
    summary.to_csv(out_dir / "coarse_region_candidate_dataset_summary.csv", index=False)

    print("\nSaved candidate dataset:", cand_path)
    print(summary)

    print("\nStatus counts:")
    print(status_df["status"].value_counts())


if __name__ == "__main__":
    main()
