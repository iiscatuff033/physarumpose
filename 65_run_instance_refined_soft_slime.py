import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.stage5h_common import (
    TARGET_TO_ID,
    ANCHOR_NAMES,
    TARGET_TO_ANCHOR_IDXS,
    build_teacher_anchor_array,
    dist2d,
)
from src.teacher_model import TeacherAnchorSoftSlimeModel


class FullImageRefinedAnchorDataset(Dataset):
    def __init__(self, gated_csv, refined_csv, image_dir, input_size=256):
        self.image_dir = Path(image_dir)
        self.input_size = int(input_size)

        gated = pd.read_csv(gated_csv)
        refined = pd.read_csv(refined_csv)

        keep = [
            "sample_id", "refined_anchor_a_x", "refined_anchor_a_y", "refined_anchor_a_conf",
            "refined_anchor_b_x", "refined_anchor_b_y", "refined_anchor_b_conf",
            "refined_region_x", "refined_region_y", "refined_region_conf",
            "refined_anchor_pair_mean_err_px", "teacher_anchor_pair_mean_err_px",
        ]
        keep = [c for c in keep if c in refined.columns]
        self.df = gated.merge(refined[keep], on="sample_id", how="left")

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_name):
        path = self.image_dir / str(image_name)
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        return img_t, w, h

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_t, img_w, img_h = self._load_image(row["occluded_image_name"])

        anchors, conf = build_teacher_anchor_array(row, img_w, img_h)

        target_name = row["target_name"]
        ia, ib = TARGET_TO_ANCHOR_IDXS[target_name]

        # Replace only the relevant anchor pair with crop-refined anchors.
        ra_x = row.get("refined_anchor_a_x", np.nan)
        ra_y = row.get("refined_anchor_a_y", np.nan)
        rb_x = row.get("refined_anchor_b_x", np.nan)
        rb_y = row.get("refined_anchor_b_y", np.nan)

        if np.isfinite(ra_x) and np.isfinite(ra_y):
            anchors[ia, 0] = np.clip(ra_x / max(img_w, 1), 0.0, 1.0)
            anchors[ia, 1] = np.clip(ra_y / max(img_h, 1), 0.0, 1.0)
            conf[ia] = max(conf[ia], row.get("refined_anchor_a_conf", 1.0))

        if np.isfinite(rb_x) and np.isfinite(rb_y):
            anchors[ib, 0] = np.clip(rb_x / max(img_w, 1), 0.0, 1.0)
            anchors[ib, 1] = np.clip(rb_y / max(img_h, 1), 0.0, 1.0)
            conf[ib] = max(conf[ib], row.get("refined_anchor_b_conf", 1.0))

        target_xy = np.array([
            row["target_orig_x"] / max(img_w, 1),
            row["target_orig_y"] / max(img_h, 1),
        ], dtype=np.float32)

        meta = row.to_dict()
        meta["img_w"] = float(img_w)
        meta["img_h"] = float(img_h)

        return {
            "image": img_t,
            "anchor_xy": torch.tensor(anchors, dtype=torch.float32),
            "anchor_conf": torch.tensor(conf, dtype=torch.float32),
            "target_xy": torch.tensor(target_xy, dtype=torch.float32),
            "target_id": torch.tensor(TARGET_TO_ID[target_name], dtype=torch.long),
            "meta": meta,
        }


def collate_fn(batch):
    out = {}
    for k in ["image", "anchor_xy", "anchor_conf", "target_xy", "target_id"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["meta"] = [b["meta"] for b in batch]
    return out


def px_error(pred_xy_norm, target_xy_norm, img_w, img_h):
    px = float(pred_xy_norm[0] * img_w)
    py = float(pred_xy_norm[1] * img_h)
    gx = float(target_xy_norm[0] * img_w)
    gy = float(target_xy_norm[1] * img_h)
    return dist2d(px, py, gx, gy), px, py


def make_topk_rows(out, batch, k=5):
    cand = out["candidate_xy"].detach().cpu().numpy()
    probs = out["candidate_probs"].detach().cpu().numpy()
    scores = out["candidate_scores"].detach().cpu().numpy()
    target_xy = batch["target_xy"].cpu().numpy()

    rows = []
    for b, meta in enumerate(batch["meta"]):
        img_w = meta["img_w"]
        img_h = meta["img_h"]
        idx = np.argsort(-probs[b])[:k]

        ax = float(out["path_anchor_a"][b, 0].detach().cpu().numpy() * img_w)
        ay = float(out["path_anchor_a"][b, 1].detach().cpu().numpy() * img_h)
        bx = float(out["path_anchor_b"][b, 0].detach().cpu().numpy() * img_w)
        by = float(out["path_anchor_b"][b, 1].detach().cpu().numpy() * img_h)

        for rank, j in enumerate(idx, start=1):
            x = float(cand[b, j, 0] * img_w)
            y = float(cand[b, j, 1] * img_h)
            gx = float(target_xy[b, 0] * img_w)
            gy = float(target_xy[b, 1] * img_h)
            err = dist2d(x, y, gx, gy)
            rows.append({
                "sample_id": meta["sample_id"],
                "image_name": meta.get("image_name", ""),
                "occluded_image_name": meta["occluded_image_name"],
                "target_name": meta["target_name"],
                "rank": rank,
                "candidate_index": int(j),
                "candidate_x": x,
                "candidate_y": y,
                "candidate_error_px": err,
                "candidate_prob": float(probs[b, j]),
                "candidate_score": float(scores[b, j]),
                "anchor1_x": ax,
                "anchor1_y": ay,
                "anchor2_x": bx,
                "anchor2_y": by,
            })
    return rows


def summarize(df):
    out = {
        "count": len(df),
        "instance_refined_gated_mean_px": df["instance_refined_gated_error_px"].mean(),
        "instance_refined_gated_median_px": df["instance_refined_gated_error_px"].median(),
        "instance_refined_top1_mean_px": df["instance_refined_top1_error_px"].mean(),
        "instance_refined_top1_median_px": df["instance_refined_top1_error_px"].median(),
        "instance_refined_soft_mean_px": df["instance_refined_soft_error_px"].mean(),
        "instance_refined_region_mean_px": df["instance_refined_region_error_px"].mean(),
        "previous_gated_mean_px": df["gated_error_px"].mean() if "gated_error_px" in df.columns else np.nan,
        "previous_top1_mean_px": df["top1_error_px"].mean() if "top1_error_px" in df.columns else np.nan,
        "top5_oracle_mean_px": df["instance_refined_top5_oracle_error_px"].mean(),
        "refined_win_vs_previous_gated": (df["instance_refined_gated_error_px"] < df["gated_error_px"]).mean() if "gated_error_px" in df.columns else np.nan,
    }

    if "yolo_target_error_px" in df.columns:
        out["yolo_mean_px"] = df["yolo_target_error_px"].mean()
        out["refined_win_vs_yolo"] = (df["instance_refined_gated_error_px"] < df["yolo_target_error_px"]).mean()

    if "suspicious_case" in df.columns:
        out["suspicious_rate"] = df["suspicious_case"].mean()

    return out


def make_error_bins(df):
    if "yolo_target_error_px" not in df.columns:
        return pd.DataFrame()

    bins = [0, 20, 40, 60, 80, 100, 150, 999999]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"]

    tmp = df.copy()
    tmp["yolo_error_bin"] = pd.cut(tmp["yolo_target_error_px"], bins=bins, labels=labels, include_lowest=True, right=False)
    return tmp.groupby("yolo_error_bin", observed=False).agg(
        count=("instance_refined_gated_error_px", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        previous_gated_mean_px=("gated_error_px", "mean"),
        instance_refined_gated_mean_px=("instance_refined_gated_error_px", "mean"),
        instance_refined_top1_mean_px=("instance_refined_top1_error_px", "mean"),
        instance_refined_region_mean_px=("instance_refined_region_error_px", "mean"),
        refined_win_vs_yolo=("instance_refined_better_than_yolo", "mean"),
    ).reset_index()


def apply_simple_gate(row):
    # Uses refined top1 unless suspicious relative to region or anchors are extreme.
    use_top1 = True
    if row["instance_refined_anchor_dist_px"] < 10:
        use_top1 = False
    if row["instance_refined_anchor_dist_px"] > 300:
        use_top1 = False
    if row["instance_refined_top1_region_dist_px"] > 100:
        use_top1 = False

    if use_top1:
        return "refined_top1_slime_path", row["instance_refined_top1_x"], row["instance_refined_top1_y"], row["instance_refined_top1_error_px"]
    else:
        return "region_fallback", row["instance_refined_region_x"], row["instance_refined_region_y"], row["instance_refined_region_error_px"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated_csv", type=str, required=True)
    parser.add_argument("--refined_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--stage5fb_model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/stage5h_instance_refined_soft_slime")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = FullImageRefinedAnchorDataset(
        gated_csv=args.gated_csv,
        refined_csv=args.refined_csv,
        image_dir=args.image_dir,
        input_size=args.input_size,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = TeacherAnchorSoftSlimeModel().to(device)
    ckpt = Path(args.stage5fb_model_dir) / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    pred_rows = []
    topk_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running instance-refined soft-slime"):
            image = batch["image"].to(device)
            target_id = batch["target_id"].to(device)
            anchor_xy = batch["anchor_xy"].to(device)
            anchor_conf = batch["anchor_conf"].to(device)

            out = model(image, target_id, anchor_xy, anchor_conf)

            soft_xy = out["soft_pred_xy"].cpu().numpy()
            region_xy = out["region_xy_all"].cpu().numpy()
            target_xy = batch["target_xy"].cpu().numpy()
            tids = batch["target_id"].cpu().numpy()

            topk = make_topk_rows(out, batch, k=5)
            topk_rows.extend(topk)

            # group topk by sample_id
            by_sample = {}
            for r in topk:
                by_sample.setdefault(r["sample_id"], []).append(r)

            for b, meta in enumerate(batch["meta"]):
                img_w = meta["img_w"]
                img_h = meta["img_h"]
                tid = int(tids[b])

                soft_err, soft_x, soft_y = px_error(soft_xy[b], target_xy[b], img_w, img_h)
                reg_err, reg_x, reg_y = px_error(region_xy[b, tid], target_xy[b], img_w, img_h)

                one = sorted(by_sample[meta["sample_id"]], key=lambda r: r["rank"])
                top1 = one[0]
                top3_oracle = min(r["candidate_error_px"] for r in one[:3])
                top5_oracle = min(r["candidate_error_px"] for r in one[:5])

                a_dist = dist2d(top1["anchor1_x"], top1["anchor1_y"], top1["anchor2_x"], top1["anchor2_y"])
                top1_region_dist = dist2d(top1["candidate_x"], top1["candidate_y"], reg_x, reg_y)

                row = dict(meta)
                row.update({
                    "instance_refined_soft_x": soft_x,
                    "instance_refined_soft_y": soft_y,
                    "instance_refined_soft_error_px": soft_err,

                    "instance_refined_region_x": reg_x,
                    "instance_refined_region_y": reg_y,
                    "instance_refined_region_error_px": reg_err,

                    "instance_refined_top1_x": top1["candidate_x"],
                    "instance_refined_top1_y": top1["candidate_y"],
                    "instance_refined_top1_error_px": top1["candidate_error_px"],

                    "instance_refined_top3_oracle_error_px": top3_oracle,
                    "instance_refined_top5_oracle_error_px": top5_oracle,
                    "instance_refined_anchor1_x": top1["anchor1_x"],
                    "instance_refined_anchor1_y": top1["anchor1_y"],
                    "instance_refined_anchor2_x": top1["anchor2_x"],
                    "instance_refined_anchor2_y": top1["anchor2_y"],
                    "instance_refined_anchor_dist_px": a_dist,
                    "instance_refined_top1_region_dist_px": top1_region_dist,
                })

                source, gx, gy, ge = apply_simple_gate(row)
                row["instance_refined_gate_source"] = source
                row["instance_refined_gated_x"] = gx
                row["instance_refined_gated_y"] = gy
                row["instance_refined_gated_error_px"] = ge

                if "yolo_target_error_px" in row:
                    row["instance_refined_better_than_yolo"] = ge < row["yolo_target_error_px"]
                if "gated_error_px" in row:
                    row["instance_refined_better_than_previous_gated"] = ge < row["gated_error_px"]

                pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    topk_df = pd.DataFrame(topk_rows)

    pred_df.to_csv(out_dir / "instance_refined_predictions.csv", index=False)
    topk_df.to_csv(out_dir / "instance_refined_topk_predictions.csv", index=False)

    summary = pd.DataFrame([summarize(pred_df)])
    summary.to_csv(out_dir / "instance_refined_summary.csv", index=False)

    by_target = []
    for target_name, g in pred_df.groupby("target_name"):
        s = summarize(g)
        s["target_name"] = target_name
        by_target.append(s)
    pd.DataFrame(by_target).to_csv(out_dir / "instance_refined_summary_by_target.csv", index=False)

    bins = make_error_bins(pred_df)
    if len(bins):
        bins.to_csv(out_dir / "instance_refined_error_bins.csv", index=False)

    print("\nSummary:")
    print(summary.T)
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
