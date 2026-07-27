import math
from pathlib import Path

import numpy as np
import pandas as pd


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def point_error(x1, y1, x2, y2):
    vals = [x1, y1, x2, y2]
    if any(pd.isna(v) for v in vals):
        return np.nan
    return float(np.sqrt((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2))


def softmax(scores, temperature=0.10):
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return np.asarray([], dtype=np.float64)

    temperature = max(float(temperature), 1e-6)
    z = scores / temperature
    z = z - np.max(z)
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def select_diverse_topk(group, top_k=5, nms_dist=12.0, score_col="learned_score"):
    """
    Select top-K hypotheses while avoiding near-duplicate candidates.
    This keeps multiple actual paths rather than 5 almost identical points.
    """
    g = group.sort_values(score_col, ascending=False).reset_index(drop=True)

    selected = []

    for _, row in g.iterrows():
        x = safe_float(row["candidate_x"])
        y = safe_float(row["candidate_y"])

        if pd.isna(x) or pd.isna(y):
            continue

        too_close = False
        for old in selected:
            ox = safe_float(old["candidate_x"])
            oy = safe_float(old["candidate_y"])
            d = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if d < nms_dist:
                too_close = True
                break

        if not too_close:
            selected.append(row)

        if len(selected) >= top_k:
            break

    # Fallback: if NMS is too strict, fill remaining with raw top predictions.
    if len(selected) < top_k:
        used_ids = set(int(r["candidate_id"]) for r in selected)
        for _, row in g.iterrows():
            cid = int(row["candidate_id"])
            if cid not in used_ids:
                selected.append(row)
                used_ids.add(cid)
            if len(selected) >= top_k:
                break

    return pd.DataFrame(selected).reset_index(drop=True)


def mean_pairwise_distance(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0

    ds = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ds.append(float(np.linalg.norm(pts[i] - pts[j])))

    return float(np.mean(ds)) if ds else 0.0


def min_error_topk(rows, k):
    sub = rows.head(k)
    if len(sub) == 0:
        return np.nan
    return float(sub["candidate_error_px"].min())


def hit_topk(rows, k, thresh):
    err = min_error_topk(rows, k)
    if pd.isna(err):
        return np.nan
    return 1.0 if err <= thresh else 0.0


def summarize_numeric(df, prefix):
    return {
        f"{prefix}_mean_px": df[f"{prefix}_error_px"].mean(),
        f"{prefix}_median_px": df[f"{prefix}_error_px"].median(),
    }


def add_yolo_bins(best_df):
    if "yolo_target_error_px" not in best_df.columns:
        return pd.DataFrame()

    bins = [0, 20, 40, 60, 80, 100, 150, 999999]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"]

    df = best_df.copy()
    df["yolo_error_bin"] = pd.cut(
        df["yolo_target_error_px"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    return df.groupby("yolo_error_bin", observed=False).agg(
        count=("sample_id", "count"),
        yolo_mean_px=("yolo_target_error_px", "mean"),
        top1_mean_px=("top1_error_px", "mean"),
        top3_oracle_mean_px=("top3_oracle_error_px", "mean"),
        top5_oracle_mean_px=("top5_oracle_error_px", "mean"),
        top3_hit20=("top3_hit_20px", "mean"),
        top5_hit20=("top5_hit_20px", "mean"),
        top1_win_vs_yolo=("top1_better_than_yolo", "mean"),
    ).reset_index()


def group_summary(wide_df):
    data = {
        "count": len(wide_df),

        "top1_mean_px": wide_df["top1_error_px"].mean(),
        "top1_median_px": wide_df["top1_error_px"].median(),

        "top3_oracle_mean_px": wide_df["top3_oracle_error_px"].mean(),
        "top3_oracle_median_px": wide_df["top3_oracle_error_px"].median(),

        "top5_oracle_mean_px": wide_df["top5_oracle_error_px"].mean(),
        "top5_oracle_median_px": wide_df["top5_oracle_error_px"].median(),

        "top3_hit_10px": wide_df["top3_hit_10px"].mean(),
        "top3_hit_20px": wide_df["top3_hit_20px"].mean(),
        "top3_hit_30px": wide_df["top3_hit_30px"].mean(),

        "top5_hit_10px": wide_df["top5_hit_10px"].mean(),
        "top5_hit_20px": wide_df["top5_hit_20px"].mean(),
        "top5_hit_30px": wide_df["top5_hit_30px"].mean(),

        "top5_diversity_mean_px": wide_df["top5_diversity_px"].mean(),
        "top3_diversity_mean_px": wide_df["top3_diversity_px"].mean(),
        "top1_probability_mean": wide_df["top1_probability"].mean(),
    }

    optional_error_cols = [
        "yolo_target_error_px",
        "region_error_px",
        "coarse_region_error_px",
        "nearest_coarse_error_px",
        "heuristic_error_px",
        "oracle_error_px",
    ]

    for col in optional_error_cols:
        if col in wide_df.columns:
            name = col.replace("_error_px", "")
            data[f"{name}_mean_px"] = wide_df[col].mean()
            data[f"{name}_median_px"] = wide_df[col].median()

    if "yolo_target_error_px" in wide_df.columns:
        data["top1_win_vs_yolo"] = wide_df["top1_better_than_yolo"].mean()
        data["top3_oracle_win_vs_yolo"] = (wide_df["top3_oracle_error_px"] < wide_df["yolo_target_error_px"]).mean()
        data["top5_oracle_win_vs_yolo"] = (wide_df["top5_oracle_error_px"] < wide_df["yolo_target_error_px"]).mean()

    if "coarse_region_error_px" in wide_df.columns:
        data["top1_win_vs_coarse_region"] = (wide_df["top1_error_px"] < wide_df["coarse_region_error_px"]).mean()
        data["top3_oracle_win_vs_coarse_region"] = (wide_df["top3_oracle_error_px"] < wide_df["coarse_region_error_px"]).mean()
        data["top5_oracle_win_vs_coarse_region"] = (wide_df["top5_oracle_error_px"] < wide_df["coarse_region_error_px"]).mean()

    return data
