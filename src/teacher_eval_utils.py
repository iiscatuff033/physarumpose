import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def px_error_norm(pred_xy_norm, gt_xy_norm, img_w, img_h):
    px = pred_xy_norm[0] * img_w
    py = pred_xy_norm[1] * img_h
    gx = gt_xy_norm[0] * img_w
    gy = gt_xy_norm[1] * img_h
    err = float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
    return err, float(px), float(py)


def summarize_predictions(df):
    out = {
        "count": len(df),
        "teacher_anchor_soft_mean_px": df["teacher_anchor_soft_error_px"].mean(),
        "teacher_anchor_soft_median_px": df["teacher_anchor_soft_error_px"].median(),
        "region_mean_px": df["teacher_region_error_px"].mean(),
        "region_median_px": df["teacher_region_error_px"].median(),
        "top1_candidate_mean_px": df["top1_candidate_error_px"].mean(),
        "top1_candidate_median_px": df["top1_candidate_error_px"].median(),
        "top3_oracle_mean_px": df["top3_oracle_error_px"].mean(),
        "top5_oracle_mean_px": df["top5_oracle_error_px"].mean(),
        "mean_teacher_anchor_err": df["mean_anchor_err"].mean() if "mean_anchor_err" in df.columns else np.nan,
    }

    if "yolo_target_error_px" in df.columns:
        out["yolo_mean_px"] = df["yolo_target_error_px"].mean()
        out["yolo_median_px"] = df["yolo_target_error_px"].median()
        out["teacher_soft_win_vs_yolo"] = (df["teacher_anchor_soft_error_px"] < df["yolo_target_error_px"]).mean()

    return out


def make_error_bins(df):
    if "yolo_target_error_px" not in df.columns:
        return pd.DataFrame()

    bins = [0, 20, 40, 60, 80, 100, 150, 999999]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"]

    out = df.copy()
    out["yolo_error_bin"] = pd.cut(
        out["yolo_target_error_px"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    return out.groupby("yolo_error_bin", observed=False).agg(
        count=("teacher_anchor_soft_error_px", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        teacher_anchor_soft_mean_px=("teacher_anchor_soft_error_px", "mean"),
        teacher_region_mean_px=("teacher_region_error_px", "mean"),
        top1_candidate_mean_px=("top1_candidate_error_px", "mean"),
        top3_oracle_mean_px=("top3_oracle_error_px", "mean"),
        top5_oracle_mean_px=("top5_oracle_error_px", "mean"),
        teacher_soft_win_vs_yolo=("teacher_soft_better_than_yolo", "mean"),
    ).reset_index()


def candidate_topk_rows(out, target_xy, img_w, img_h, k=5):
    cand = out["candidate_xy"].detach().cpu().numpy()
    probs = out["candidate_probs"].detach().cpu().numpy()
    scores = out["candidate_scores"].detach().cpu().numpy()

    rows = []
    for b in range(cand.shape[0]):
        idx = np.argsort(-probs[b])[:k]
        for rank, j in enumerate(idx, start=1):
            x = float(cand[b, j, 0] * img_w[b])
            y = float(cand[b, j, 1] * img_h[b])
            gx = float(target_xy[b, 0] * img_w[b])
            gy = float(target_xy[b, 1] * img_h[b])
            err = float(np.sqrt((x - gx) ** 2 + (y - gy) ** 2))
            rows.append({
                "batch_index": b,
                "rank": rank,
                "candidate_index": int(j),
                "candidate_x": x,
                "candidate_y": y,
                "candidate_error_px": err,
                "candidate_prob": float(probs[b, j]),
                "candidate_score": float(scores[b, j]),
            })
    return rows
