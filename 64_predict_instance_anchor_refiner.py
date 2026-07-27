import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.stage5h_common import (
    merge_crop_and_strong,
    dist2d,
    get_teacher_anchor_full,
    TARGET_TO_ANCHOR_NAMES,
)
from src.anchor_refiner_dataset import InstanceAnchorRefinerDataset
from src.anchor_refiner_model import InstanceAnchorRefiner, soft_argmax_2d


def collate_fn(batch):
    out = {}
    out["image"] = torch.stack([b["image"] for b in batch], dim=0)
    out["xy"] = torch.stack([b["xy"] for b in batch], dim=0)
    out["heatmaps"] = torch.stack([b["heatmaps"] for b in batch], dim=0)
    out["target_id"] = torch.stack([b["target_id"] for b in batch], dim=0)
    out["meta"] = [b["meta"] for b in batch]
    return out


def crop_norm_to_full(xn, yn, meta):
    x1 = meta["crop_x1"]
    y1 = meta["crop_y1"]
    x2 = meta["crop_x2"]
    y2 = meta["crop_y2"]
    x = x1 + float(xn) * max(x2 - x1, 1.0)
    y = y1 + float(yn) * max(y2 - y1, 1.0)
    return x, y


def summarize(df):
    out = {
        "count": len(df),
        "refined_anchor_pair_mean_err_px": df["refined_anchor_pair_mean_err_px"].mean(),
        "refined_anchor_pair_median_err_px": df["refined_anchor_pair_mean_err_px"].median(),
        "teacher_anchor_pair_mean_err_px": df["teacher_anchor_pair_mean_err_px"].mean(),
        "teacher_anchor_pair_median_err_px": df["teacher_anchor_pair_mean_err_px"].median(),
        "refined_better_than_teacher_rate": (df["refined_anchor_pair_mean_err_px"] < df["teacher_anchor_pair_mean_err_px"]).mean(),
        "refined_region_mean_err_px": df["refined_region_error_px"].mean(),
        "refined_region_median_err_px": df["refined_region_error_px"].median(),
        "mean_anchor_conf": df[["refined_anchor_a_conf", "refined_anchor_b_conf"]].mean(axis=1).mean(),
    }
    if "suspicious_case" in df.columns:
        out["suspicious_rate"] = df["suspicious_case"].mean()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_csv", type=str, required=True)
    parser.add_argument("--strong_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5h_instance_anchor_refiner/predictions")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merge_crop_and_strong(args.crop_csv, args.strong_csv)

    ds = InstanceAnchorRefinerDataset(
        df,
        image_dir=args.image_dir,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        augment=False,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = InstanceAnchorRefiner(out_channels=3).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "best_model.pt", map_location=device))
    model.eval()

    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting instance anchors"):
            image = batch["image"].to(device)
            logits = model(image)
            xy_pred, conf = soft_argmax_2d(logits, temperature=0.05)
            xy_pred = xy_pred.cpu().numpy()
            conf = conf.cpu().numpy()

            for b, meta in enumerate(batch["meta"]):
                target_name = meta["target_name"]
                a_name, b_name = TARGET_TO_ANCHOR_NAMES[target_name]

                pa_x, pa_y = crop_norm_to_full(xy_pred[b, 0, 0], xy_pred[b, 0, 1], meta)
                pb_x, pb_y = crop_norm_to_full(xy_pred[b, 1, 0], xy_pred[b, 1, 1], meta)
                pr_x, pr_y = crop_norm_to_full(xy_pred[b, 2, 0], xy_pred[b, 2, 1], meta)

                gt_ax = meta["gt_anchor_a_x"]
                gt_ay = meta["gt_anchor_a_y"]
                gt_bx = meta["gt_anchor_b_x"]
                gt_by = meta["gt_anchor_b_y"]
                gt_tx = meta["target_orig_x"]
                gt_ty = meta["target_orig_y"]

                ta_x, ta_y, ta_c = get_teacher_anchor_full(pd.Series(meta), a_name)
                tb_x, tb_y, tb_c = get_teacher_anchor_full(pd.Series(meta), b_name)

                # meta from dataset contains only generic teacher_ columns if crop CSV passed has them.
                # If generic lookup failed, try explicit values from crop CSV row via merge is handled in full below.
                # Here fallback to predicted gate anchors for selected pair when teacher names missing in meta.
                if not np.isfinite(ta_x):
                    ta_x, ta_y, ta_c = np.nan, np.nan, np.nan
                if not np.isfinite(tb_x):
                    tb_x, tb_y, tb_c = np.nan, np.nan, np.nan

                refined_a_err = dist2d(pa_x, pa_y, gt_ax, gt_ay)
                refined_b_err = dist2d(pb_x, pb_y, gt_bx, gt_by)
                teacher_a_err = dist2d(ta_x, ta_y, gt_ax, gt_ay)
                teacher_b_err = dist2d(tb_x, tb_y, gt_bx, gt_by)
                refined_region_err = dist2d(pr_x, pr_y, gt_tx, gt_ty)

                rows.append({
                    "sample_id": meta["sample_id"],
                    "image_name": meta["image_name"],
                    "occluded_image_name": meta["occluded_image_name"],
                    "target_name": target_name,
                    "anchor_a_name": a_name,
                    "anchor_b_name": b_name,

                    "refined_anchor_a_x": pa_x,
                    "refined_anchor_a_y": pa_y,
                    "refined_anchor_a_conf": float(conf[b, 0]),
                    "refined_anchor_b_x": pb_x,
                    "refined_anchor_b_y": pb_y,
                    "refined_anchor_b_conf": float(conf[b, 1]),
                    "refined_region_x": pr_x,
                    "refined_region_y": pr_y,
                    "refined_region_conf": float(conf[b, 2]),

                    "gt_anchor_a_x": gt_ax,
                    "gt_anchor_a_y": gt_ay,
                    "gt_anchor_b_x": gt_bx,
                    "gt_anchor_b_y": gt_by,
                    "target_orig_x": gt_tx,
                    "target_orig_y": gt_ty,

                    "refined_anchor_a_err_px": refined_a_err,
                    "refined_anchor_b_err_px": refined_b_err,
                    "refined_anchor_pair_mean_err_px": np.nanmean([refined_a_err, refined_b_err]),
                    "teacher_anchor_a_err_px": teacher_a_err,
                    "teacher_anchor_b_err_px": teacher_b_err,
                    "teacher_anchor_pair_mean_err_px": np.nanmean([teacher_a_err, teacher_b_err]),
                    "refined_region_error_px": refined_region_err,
                    "suspicious_case": meta.get("suspicious_case", 0),
                    "previous_gated_error_px": meta.get("gated_error_px", np.nan),
                })

    out = pd.DataFrame(rows)

    # Add teacher errors again from original df more reliably.
    df_small = df[["sample_id"] + [c for c in df.columns if c.startswith("teacher_")]].copy()
    out = out.merge(df_small, on="sample_id", how="left", suffixes=("", "_from_crop"))
    final_teacher_pair = []
    for _, r in out.iterrows():
        a_name, b_name = TARGET_TO_ANCHOR_NAMES[r["target_name"]]
        ta_x = r.get(f"teacher_{a_name}_x", np.nan)
        ta_y = r.get(f"teacher_{a_name}_y", np.nan)
        tb_x = r.get(f"teacher_{b_name}_x", np.nan)
        tb_y = r.get(f"teacher_{b_name}_y", np.nan)
        ea = dist2d(ta_x, ta_y, r["gt_anchor_a_x"], r["gt_anchor_a_y"])
        eb = dist2d(tb_x, tb_y, r["gt_anchor_b_x"], r["gt_anchor_b_y"])
        final_teacher_pair.append(np.nanmean([ea, eb]))
    out["teacher_anchor_pair_mean_err_px"] = final_teacher_pair

    out.to_csv(out_dir / "refined_anchor_predictions.csv", index=False)

    summary = pd.DataFrame([summarize(out)])
    summary.to_csv(out_dir / "refined_anchor_summary.csv", index=False)

    by_target = []
    for t, g in out.groupby("target_name"):
        s = summarize(g)
        s["target_name"] = t
        by_target.append(s)
    pd.DataFrame(by_target).to_csv(out_dir / "refined_anchor_summary_by_target.csv", index=False)

    print("\nSummary:")
    print(summary.T)
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
