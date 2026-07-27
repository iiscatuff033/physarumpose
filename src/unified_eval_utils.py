import numpy as np
import pandas as pd


def dist2d(x1, y1, x2, y2):
    vals = [x1, y1, x2, y2]
    if any(pd.isna(v) for v in vals):
        return np.nan
    return float(np.sqrt((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2))


def crop_norm_to_full(xn, yn, meta):
    x = float(meta["crop_x1"]) + float(xn) * max(float(meta["crop_x2"]) - float(meta["crop_x1"]), 1.0)
    y = float(meta["crop_y1"]) + float(yn) * max(float(meta["crop_y2"]) - float(meta["crop_y1"]), 1.0)
    return x, y


def px_error_from_norm(xy, meta):
    x, y = crop_norm_to_full(xy[0], xy[1], meta)
    err = dist2d(x, y, meta["target_orig_x"], meta["target_orig_y"])
    return err, x, y


def point_error(pred_x, pred_y, gt_x, gt_y):
    return dist2d(pred_x, pred_y, gt_x, gt_y)


def topk_rows_from_output(out, batch, k=5):
    cand = out["candidate_xy"].detach().cpu().numpy()
    probs = out["candidate_probs"].detach().cpu().numpy()
    scores = out["candidate_scores"].detach().cpu().numpy()

    rows = []
    for b, meta in enumerate(batch["meta"]):
        idx = np.argsort(-probs[b])[:k]

        ax, ay = crop_norm_to_full(out["path_anchor_a"][b, 0].detach().cpu().numpy(), out["path_anchor_a"][b, 1].detach().cpu().numpy(), meta)
        bx, by = crop_norm_to_full(out["path_anchor_b"][b, 0].detach().cpu().numpy(), out["path_anchor_b"][b, 1].detach().cpu().numpy(), meta)

        for rank, j in enumerate(idx, start=1):
            x, y = crop_norm_to_full(cand[b, j, 0], cand[b, j, 1], meta)
            err = dist2d(x, y, meta["target_orig_x"], meta["target_orig_y"])
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


def summarize_predictions(df):
    out = {
        "count": len(df),
        "unified_soft_mean_px": df["unified_soft_error_px"].mean(),
        "unified_soft_median_px": df["unified_soft_error_px"].median(),
        "unified_top1_mean_px": df["unified_top1_error_px"].mean(),
        "unified_top1_median_px": df["unified_top1_error_px"].median(),
        "unified_region_mean_px": df["unified_region_error_px"].mean(),
        "unified_region_median_px": df["unified_region_error_px"].median(),
        "unified_anchor_pair_mean_px": df["unified_anchor_pair_mean_error_px"].mean(),
        "unified_anchor_pair_median_px": df["unified_anchor_pair_mean_error_px"].median(),
        "top3_oracle_mean_px": df["unified_top3_oracle_error_px"].mean(),
        "top5_oracle_mean_px": df["unified_top5_oracle_error_px"].mean(),
        "soft_win_vs_top1_rate": (df["unified_soft_error_px"] < df["unified_top1_error_px"]).mean(),
        "top1_win_vs_soft_rate": (df["unified_top1_error_px"] < df["unified_soft_error_px"]).mean(),
    }

    if "yolo_target_error_px" in df.columns:
        out["yolo_mean_px"] = df["yolo_target_error_px"].mean()
        out["yolo_median_px"] = df["yolo_target_error_px"].median()
        out["soft_win_vs_yolo_rate"] = (df["unified_soft_error_px"] < df["yolo_target_error_px"]).mean()
        out["top1_win_vs_yolo_rate"] = (df["unified_top1_error_px"] < df["yolo_target_error_px"]).mean()

    if "gated_error_px" in df.columns:
        out["previous_gated_mean_px"] = df["gated_error_px"].mean()
        out["previous_gated_median_px"] = df["gated_error_px"].median()
        out["soft_win_vs_previous_gated_rate"] = (df["unified_soft_error_px"] < df["gated_error_px"]).mean()
        out["top1_win_vs_previous_gated_rate"] = (df["unified_top1_error_px"] < df["gated_error_px"]).mean()

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

    agg = {
        "count": ("unified_soft_error_px", "count"),
        "yolo_mean_px": ("yolo_target_error_px", "mean"),
        "unified_soft_mean_px": ("unified_soft_error_px", "mean"),
        "unified_top1_mean_px": ("unified_top1_error_px", "mean"),
        "unified_region_mean_px": ("unified_region_error_px", "mean"),
        "unified_anchor_pair_mean_px": ("unified_anchor_pair_mean_error_px", "mean"),
        "top5_oracle_mean_px": ("unified_top5_oracle_error_px", "mean"),
    }

    if "gated_error_px" in df.columns:
        agg["previous_gated_mean_px"] = ("gated_error_px", "mean")

    return tmp.groupby("yolo_error_bin", observed=False).agg(**agg).reset_index()
