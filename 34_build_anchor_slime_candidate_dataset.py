import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.learned_slime_utils import (
    FEATURE_NAMES,
    build_candidate_table_for_row,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5_learned_slime_candidates")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--min_anchor_conf", type=float, default=0.03)
    parser.add_argument("--no_edges", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    all_tables = []
    status_rows = []

    for sample_id, row in tqdm(df.iterrows(), total=len(df), desc="Building learned-slime candidate dataset"):
        img_path = image_dir / row["occluded_image_name"]
        status = "ok"

        if not img_path.exists():
            status = "missing_image"
            status_rows.append({"sample_id": sample_id, "status": status, "num_candidates": 0})
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            status = "bad_image"
            status_rows.append({"sample_id": sample_id, "status": status, "num_candidates": 0})
            continue

        cand_df, status = build_candidate_table_for_row(
            row=row,
            image_bgr=image_bgr,
            sample_id=sample_id,
            use_edges=not args.no_edges,
        )

        if status == "ok" and len(cand_df) > 0:
            # keep anchors with very low confidence if user wants; default only filters obvious failures
            cand_df = cand_df[cand_df["anchor_min_conf"] >= args.min_anchor_conf].reset_index(drop=True)
            if len(cand_df) > 0:
                all_tables.append(cand_df)
            else:
                status = "low_anchor_conf"

        status_rows.append({
            "sample_id": sample_id,
            "status": status,
            "num_candidates": 0 if cand_df is None else len(cand_df),
            "target_name": row.get("target_name", ""),
            "occluded_image_name": row.get("occluded_image_name", ""),
        })

    if not all_tables:
        raise RuntimeError("No candidate rows were created. Check anchor columns and image paths.")

    out_df = pd.concat(all_tables, ignore_index=True)
    out_csv = out_dir / "anchor_slime_candidate_dataset.csv"
    out_df.to_csv(out_csv, index=False)

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(out_dir / "candidate_build_status.csv", index=False)

    save_json({"feature_names": FEATURE_NAMES}, out_dir / "candidate_feature_names.json")

    summary = pd.DataFrame([{
        "input_samples": len(df),
        "valid_samples": int(out_df["sample_id"].nunique()),
        "candidate_rows": len(out_df),
        "mean_candidates_per_valid_sample": len(out_df) / max(out_df["sample_id"].nunique(), 1),
        "mean_label_quality": out_df["label_quality"].mean(),
        "good_candidate_rate": out_df["label_good"].mean(),
    }])
    summary.to_csv(out_dir / "candidate_dataset_summary.csv", index=False)

    by_target = out_df.groupby("target_name").agg(
        samples=("sample_id", "nunique"),
        candidates=("candidate_id", "count"),
        mean_best_error_px=("candidate_error_px", "min"),
        good_candidate_rate=("label_good", "mean"),
    ).reset_index()
    by_target.to_csv(out_dir / "candidate_dataset_summary_by_target.csv", index=False)

    print(f"\nSaved candidate dataset: {out_csv}")
    print(f"Saved feature list: {out_dir / 'candidate_feature_names.json'}")
    print("\nSummary:")
    print(summary)
    print("\nStatus counts:")
    print(status_df["status"].value_counts())


if __name__ == "__main__":
    main()
