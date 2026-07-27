import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.e2e_dataset import E2ESoftSlimeDataset
from src.e2e_model import E2ESoftSlimeModel
from src.e2e_eval_utils import (
    load_json,
    px_error_norm,
    candidate_topk,
    summarize_predictions,
    make_error_bins,
)


def collate_fn(batch):
    out = {}
    keys = ["image", "anchor_xy", "anchor_mask", "target_xy", "target_id", "anchor_heatmaps", "target_heatmaps"]
    for k in keys:
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
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5f_e2e_soft_slime/eval")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = E2ESoftSlimeDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        max_samples=args.max_samples,
        augment=False,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = E2ESoftSlimeModel().to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "best_model.pt", map_location=device))
    model.eval()

    pred_rows = []
    topk_rows = []

    sample_counter = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating E2E soft-slime"):
            batch = move_batch(batch, device)
            out = model(batch["image"], batch["target_id"])

            soft_xy = out["soft_pred_xy"].cpu().numpy()
            region_xy_all = out["region_xy_all"].cpu().numpy()
            target_xy = batch["target_xy"].cpu().numpy()
            target_id = batch["target_id"].cpu().numpy()

            img_w = np.array([m["img_w"] for m in batch["meta"]], dtype=np.float32)
            img_h = np.array([m["img_h"] for m in batch["meta"]], dtype=np.float32)

            topk_raw = candidate_topk(
                out,
                target_xy=target_xy,
                img_w=img_w,
                img_h=img_h,
                k=5,
            )

            # Group topk raw by batch_index
            by_b = {}
            for r in topk_raw:
                by_b.setdefault(r["batch_index"], []).append(r)

            for b, meta in enumerate(batch["meta"]):
                tid = int(target_id[b])

                e2e_err, e2e_x, e2e_y = px_error_norm(soft_xy[b], target_xy[b], img_w[b], img_h[b])
                reg_err, reg_x, reg_y = px_error_norm(region_xy_all[b, tid], target_xy[b], img_w[b], img_h[b])

                one_top = by_b.get(b, [])
                top1_err = one_top[0]["candidate_error_px"] if one_top else np.nan
                top3_oracle = min([r["candidate_error_px"] for r in one_top[:3]]) if one_top else np.nan
                top5_oracle = min([r["candidate_error_px"] for r in one_top[:5]]) if one_top else np.nan

                row = {
                    "sample_id": sample_counter,
                    "image_name": meta["image_name"],
                    "occluded_image_name": meta["occluded_image_name"],
                    "target_name": meta["target_name"],
                    "target_orig_x": meta["target_orig_x"],
                    "target_orig_y": meta["target_orig_y"],

                    "e2e_soft_x": e2e_x,
                    "e2e_soft_y": e2e_y,
                    "e2e_soft_error_px": e2e_err,

                    "e2e_region_x": reg_x,
                    "e2e_region_y": reg_y,
                    "e2e_region_error_px": reg_err,

                    "top1_candidate_error_px": top1_err,
                    "top3_oracle_error_px": top3_oracle,
                    "top5_oracle_error_px": top5_oracle,
                }

                for col in [
                    "yolo_target_x", "yolo_target_y", "yolo_target_error_px",
                    "region_pred_x", "region_pred_y", "region_error_px",
                    "geometry_x", "geometry_y", "geometry_error_px",
                ]:
                    if col in meta:
                        row[col] = meta[col]

                if "yolo_target_error_px" in row:
                    row["e2e_better_than_yolo"] = row["e2e_soft_error_px"] < row["yolo_target_error_px"]

                pred_rows.append(row)

                for r in one_top:
                    rr = dict(r)
                    rr.pop("batch_index", None)
                    rr["sample_id"] = sample_counter
                    rr["image_name"] = meta["image_name"]
                    rr["occluded_image_name"] = meta["occluded_image_name"]
                    rr["target_name"] = meta["target_name"]
                    rr["target_orig_x"] = meta["target_orig_x"]
                    rr["target_orig_y"] = meta["target_orig_y"]
                    rr["anchor1_x"] = float(out["path_anchor_a"][b, 0].cpu().numpy() * img_w[b])
                    rr["anchor1_y"] = float(out["path_anchor_a"][b, 1].cpu().numpy() * img_h[b])
                    rr["anchor2_x"] = float(out["path_anchor_b"][b, 0].cpu().numpy() * img_w[b])
                    rr["anchor2_y"] = float(out["path_anchor_b"][b, 1].cpu().numpy() * img_h[b])
                    topk_rows.append(rr)

                sample_counter += 1

    pred_df = pd.DataFrame(pred_rows)
    topk_df = pd.DataFrame(topk_rows)

    pred_df.to_csv(out_dir / "e2e_predictions.csv", index=False)
    topk_df.to_csv(out_dir / "e2e_topk_predictions.csv", index=False)

    summary = pd.DataFrame([summarize_predictions(pred_df)])
    summary.to_csv(out_dir / "e2e_summary.csv", index=False)

    by_target = []
    for target_name, g in pred_df.groupby("target_name"):
        row = {"target_name": target_name}
        row.update(summarize_predictions(g))
        by_target.append(row)
    by_target = pd.DataFrame(by_target)
    by_target.to_csv(out_dir / "e2e_summary_by_target.csv", index=False)

    bins = make_error_bins(pred_df)
    if len(bins) > 0:
        bins.to_csv(out_dir / "e2e_error_bins.csv", index=False)

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
