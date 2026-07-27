import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.crop_filter_utils import (
    get_image_size,
    crop_box_from_points,
    merge_topk_if_needed,
    add_crop_quality_flags,
    summarize_crop_quality,
    point_inside_crop,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated_csv", type=str, required=True)
    parser.add_argument("--topk_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5g_target_person_crop_filter")
    parser.add_argument("--pad", type=int, default=120)
    parser.add_argument("--min_size", type=int, default=260)
    parser.add_argument("--include_gt_in_crop", action="store_true",
                        help="Use GT target point when defining crop. Good only for debug/presentation, not deployment-style.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    gated = pd.read_csv(args.gated_csv)
    topk = pd.read_csv(args.topk_csv)

    df = merge_topk_if_needed(gated, topk)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating crop metadata"):
        image_name = row["occluded_image_name"]
        image_path = image_dir / image_name

        img_w, img_h = get_image_size(image_path)
        if img_w is None:
            continue

        # Deployment-style crop points: do NOT require GT.
        pts = [
            (row.get("gate_anchor1_x"), row.get("gate_anchor1_y")),
            (row.get("gate_anchor2_x"), row.get("gate_anchor2_y")),
            (row.get("top1_x"), row.get("top1_y")),
            (row.get("teacher_region_x"), row.get("teacher_region_y")),
            (row.get("gated_x"), row.get("gated_y")),
        ]

        # Optional debug/presentation setting.
        if args.include_gt_in_crop:
            pts.append((row.get("target_orig_x"), row.get("target_orig_y")))

        x1, y1, x2, y2 = crop_box_from_points(
            pts,
            img_w=img_w,
            img_h=img_h,
            pad=args.pad,
            min_size=args.min_size,
        )

        r = row.to_dict()
        r.update({
            "img_w": img_w,
            "img_h": img_h,
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
            "crop_w": x2 - x1,
            "crop_h": y2 - y1,
            "crop_used_gt_for_box": int(args.include_gt_in_crop),
        })

        # Whether important diagnostic points lie in crop.
        for point_name, xcol, ycol in [
            ("anchor1", "gate_anchor1_x", "gate_anchor1_y"),
            ("anchor2", "gate_anchor2_x", "gate_anchor2_y"),
            ("top1", "top1_x", "top1_y"),
            ("region", "teacher_region_x", "teacher_region_y"),
            ("gated", "gated_x", "gated_y"),
            ("gt", "target_orig_x", "target_orig_y"),
        ]:
            r[f"{point_name}_inside_crop"] = int(point_inside_crop(
                row.get(xcol), row.get(ycol), x1, y1, x2, y2
            ))

        rows.append(r)

    out = pd.DataFrame(rows)
    out = add_crop_quality_flags(out)

    out.to_csv(out_dir / "crop_metadata.csv", index=False)

    summary = pd.DataFrame([summarize_crop_quality(out)])
    summary.to_csv(out_dir / "crop_quality_summary.csv", index=False)

    by_target = []
    for target_name, g in out.groupby("target_name"):
        s = summarize_crop_quality(g)
        s["target_name"] = target_name
        by_target.append(s)
    pd.DataFrame(by_target).to_csv(out_dir / "crop_quality_summary_by_target.csv", index=False)

    print("\nCrop quality summary:")
    print(summary.T)
    print("\nSaved:", out_dir / "crop_metadata.csv")


if __name__ == "__main__":
    main()
