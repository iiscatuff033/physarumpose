import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.stage6_common import (
    merge_crop_and_annotations,
    filter_valid_gt_inside_crop,
    dist2d,
)
from src.unified_dataset import UnifiedPhysarumDataset
from src.unified_model import UnifiedPhysarumPoseModel
from src.unified_eval_utils import (
    crop_norm_to_full,
    px_error_from_norm,
    topk_rows_from_output,
    summarize_predictions,
    make_error_bins,
)


def collate_fn(batch):
    out = {}
    for k in ["image", "target_id", "xy", "heatmaps"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["meta"] = [b["meta"] for b in batch]
    return out


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if k == "meta":
            out[k] = v
        else:
            out[k] = v.to(device)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_csv", type=str, required=True)
    parser.add_argument("--annotation_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage6a_unified_physarum_pose/eval")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merge_crop_and_annotations(args.crop_csv, args.annotation_csv)
    df = filter_valid_gt_inside_crop(df, margin=0.05)

    if args.max_samples is not None and args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    ds = UnifiedPhysarumDataset(
        df,
        image_dir=args.image_dir,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        augment=False,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Eval samples:", len(ds))

    model = UnifiedPhysarumPoseModel().to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "best_model.pt", map_location=device))
    model.eval()

    pred_rows = []
    topk_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating unified PhysarumPose"):
            batch = move_batch(batch, device)
            out = model(batch["image"], batch["target_id"])

            soft_xy = out["soft_pred_xy"].detach().cpu().numpy()
            heat_xy = out["heat_xy"].detach().cpu().numpy()
            heat_conf = out["heat_conf"].detach().cpu().numpy()

            batch_topk = topk_rows_from_output(out, batch, k=5)
            topk_rows.extend(batch_topk)

            by_sample = {}
            for r in batch_topk:
                by_sample.setdefault(r["sample_id"], []).append(r)

            for b, meta in enumerate(batch["meta"]):
                soft_err, soft_x, soft_y = px_error_from_norm(soft_xy[b], meta)

                anchor_a_x, anchor_a_y = crop_norm_to_full(heat_xy[b, 0, 0], heat_xy[b, 0, 1], meta)
                anchor_b_x, anchor_b_y = crop_norm_to_full(heat_xy[b, 1, 0], heat_xy[b, 1, 1], meta)
                region_x, region_y = crop_norm_to_full(heat_xy[b, 2, 0], heat_xy[b, 2, 1], meta)

                anchor_a_err = dist2d(anchor_a_x, anchor_a_y, meta["gt_anchor_a_x"], meta["gt_anchor_a_y"])
                anchor_b_err = dist2d(anchor_b_x, anchor_b_y, meta["gt_anchor_b_x"], meta["gt_anchor_b_y"])
                region_err = dist2d(region_x, region_y, meta["target_orig_x"], meta["target_orig_y"])

                one = sorted(by_sample[meta["sample_id"]], key=lambda r: r["rank"])
                top1 = one[0]
                top3_oracle = min(r["candidate_error_px"] for r in one[:3])
                top5_oracle = min(r["candidate_error_px"] for r in one[:5])

                row = dict(meta)
                row.update({
                    "unified_soft_x": soft_x,
                    "unified_soft_y": soft_y,
                    "unified_soft_error_px": soft_err,

                    "unified_anchor_a_x": anchor_a_x,
                    "unified_anchor_a_y": anchor_a_y,
                    "unified_anchor_a_conf": float(heat_conf[b, 0]),
                    "unified_anchor_a_error_px": anchor_a_err,

                    "unified_anchor_b_x": anchor_b_x,
                    "unified_anchor_b_y": anchor_b_y,
                    "unified_anchor_b_conf": float(heat_conf[b, 1]),
                    "unified_anchor_b_error_px": anchor_b_err,

                    "unified_anchor_pair_mean_error_px": np.nanmean([anchor_a_err, anchor_b_err]),

                    "unified_region_x": region_x,
                    "unified_region_y": region_y,
                    "unified_region_conf": float(heat_conf[b, 2]),
                    "unified_region_error_px": region_err,

                    "unified_top1_x": top1["candidate_x"],
                    "unified_top1_y": top1["candidate_y"],
                    "unified_top1_error_px": top1["candidate_error_px"],
                    "unified_top1_prob": top1["candidate_prob"],

                    "unified_top3_oracle_error_px": top3_oracle,
                    "unified_top5_oracle_error_px": top5_oracle,
                    "unified_anchor_dist_px": dist2d(
                        top1["anchor1_x"], top1["anchor1_y"],
                        top1["anchor2_x"], top1["anchor2_y"],
                    ),
                })

                if "yolo_target_error_px" in row:
                    row["unified_soft_better_than_yolo"] = row["unified_soft_error_px"] < row["yolo_target_error_px"]
                    row["unified_top1_better_than_yolo"] = row["unified_top1_error_px"] < row["yolo_target_error_px"]

                if "gated_error_px" in row:
                    row["unified_soft_better_than_previous_gated"] = row["unified_soft_error_px"] < row["gated_error_px"]
                    row["unified_top1_better_than_previous_gated"] = row["unified_top1_error_px"] < row["gated_error_px"]

                pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    topk_df = pd.DataFrame(topk_rows)

    pred_df.to_csv(out_dir / "unified_predictions.csv", index=False)
    topk_df.to_csv(out_dir / "unified_topk_predictions.csv", index=False)

    summary = pd.DataFrame([summarize_predictions(pred_df)])
    summary.to_csv(out_dir / "unified_summary.csv", index=False)

    by_target = []
    for target_name, g in pred_df.groupby("target_name"):
        s = summarize_predictions(g)
        s["target_name"] = target_name
        by_target.append(s)
    pd.DataFrame(by_target).to_csv(out_dir / "unified_summary_by_target.csv", index=False)

    bins = make_error_bins(pred_df)
    if len(bins):
        bins.to_csv(out_dir / "unified_error_bins.csv", index=False)

    print("\nSummary:")
    print(summary.T)
    print("\nBy target:")
    print(pd.DataFrame(by_target))
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
