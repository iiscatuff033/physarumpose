import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.pred_region_common import get_target_anchors, valid_point, save_json
from src.pred_region_slime_utils import generate_candidates, build_candidate_features, PRED_REGION_FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5c_pred_region_candidates")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--anchor_conf_thresh", type=float, default=0.10)
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

    for sample_id, row in tqdm(df.iterrows(), total=len(df), desc="Building pred-region candidates"):
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
        pred_region = None
        if "region_pred_x" in row.index and "region_pred_y" in row.index:
            pred_region = row[["region_pred_x", "region_pred_y"]].values.astype("float32")
        candidates = generate_candidates(a, b, image_bgr.shape, pred_region=pred_region)
        if len(candidates) == 0:
            status_rows.append({"sample_id": sample_id, "status": "no_candidates"})
            continue
        cand_df = build_candidate_features(row, image_bgr, candidates, a, b, ac, bc)
        if len(cand_df) == 0:
            status_rows.append({"sample_id": sample_id, "status": "empty_candidate_features"})
            continue
        all_cands.append(cand_df)
        valid_samples += 1
        status_rows.append({"sample_id": sample_id, "status": "ok", "num_candidates": len(cand_df)})

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(out_dir / "pred_region_candidate_status.csv", index=False)
    if not all_cands:
        print("No candidate rows were created.")
        print(status_df["status"].value_counts() if "status" in status_df.columns else status_df)
        return
    cand_all = pd.concat(all_cands, ignore_index=True)
    cand_path = out_dir / "pred_region_candidate_dataset.csv"
    cand_all.to_csv(cand_path, index=False)
    save_json(PRED_REGION_FEATURES, out_dir / "pred_region_feature_names.json")
    summary = pd.DataFrame([{
        "input_samples": len(df),
        "valid_samples": valid_samples,
        "candidate_rows": len(cand_all),
        "avg_candidates_per_valid_sample": len(cand_all) / max(valid_samples, 1),
        "num_features": len(PRED_REGION_FEATURES),
        "uses_gt_occlusion_box_features": 0,
        "uses_predicted_region_features": 1,
    }])
    summary.to_csv(out_dir / "pred_region_candidate_dataset_summary.csv", index=False)
    print("\nSaved candidate dataset:", cand_path)
    print(summary)
    print("\nStatus counts:")
    print(status_df["status"].value_counts())


if __name__ == "__main__":
    main()
