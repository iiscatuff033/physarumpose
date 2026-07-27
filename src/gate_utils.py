import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_TO_ANCHORS = {
    "right_elbow": ("right_shoulder", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_wrist"),
    "right_knee": ("right_hip", "right_ankle"),
    "left_knee": ("left_hip", "left_ankle"),
}


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def dist2d(x1, y1, x2, y2):
    vals = [x1, y1, x2, y2]
    if any(pd.isna(v) for v in vals):
        return np.nan
    return float(np.sqrt((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2))


def split_by_image(df, tune_ratio=0.30, seed=42):
    rng = np.random.default_rng(seed)

    if "image_name" in df.columns:
        names = df["image_name"].fillna(df["occluded_image_name"]).unique()
        key = df["image_name"].fillna(df["occluded_image_name"])
    else:
        names = df["occluded_image_name"].unique()
        key = df["occluded_image_name"]

    rng.shuffle(names)
    n_tune = max(1, int(len(names) * tune_ratio))
    tune_names = set(names[:n_tune])

    tune = df[key.isin(tune_names)].copy()
    holdout = df[~key.isin(tune_names)].copy()

    return tune.reset_index(drop=True), holdout.reset_index(drop=True)


def merge_pred_and_topk(pred_df, topk_df):
    """
    Merge Stage 5F-B prediction rows with top-1/top-2 candidate information.
    """
    pred = pred_df.copy()
    topk = topk_df.copy()

    # Rank 1
    r1 = topk[topk["rank"] == 1].copy()
    r1 = r1.rename(columns={
        "candidate_x": "top1_x",
        "candidate_y": "top1_y",
        "candidate_error_px": "top1_error_px",
        "candidate_prob": "top1_prob",
        "candidate_score": "top1_score",
        "anchor1_x": "gate_anchor1_x",
        "anchor1_y": "gate_anchor1_y",
        "anchor2_x": "gate_anchor2_x",
        "anchor2_y": "gate_anchor2_y",
    })

    keep1 = [
        "sample_id", "top1_x", "top1_y", "top1_error_px", "top1_prob", "top1_score",
        "gate_anchor1_x", "gate_anchor1_y", "gate_anchor2_x", "gate_anchor2_y",
    ]
    keep1 = [c for c in keep1 if c in r1.columns]
    r1 = r1[keep1]

    out = pred.merge(r1, on="sample_id", how="left")

    # Rank 2 for probability/score margin
    r2 = topk[topk["rank"] == 2].copy()
    r2 = r2.rename(columns={
        "candidate_prob": "top2_prob",
        "candidate_score": "top2_score",
        "candidate_error_px": "top2_error_px",
    })
    keep2 = ["sample_id", "top2_prob", "top2_score", "top2_error_px"]
    keep2 = [c for c in keep2 if c in r2.columns]
    out = out.merge(r2[keep2], on="sample_id", how="left")

    if "top1_error_px" not in out.columns and "top1_candidate_error_px" in out.columns:
        out["top1_error_px"] = out["top1_candidate_error_px"]

    if "top1_prob" not in out.columns:
        out["top1_prob"] = 1.0
    if "top2_prob" not in out.columns:
        out["top2_prob"] = 0.0
    if "top1_score" not in out.columns:
        out["top1_score"] = 0.0
    if "top2_score" not in out.columns:
        out["top2_score"] = out["top1_score"] - 1.0

    out["top_prob_margin"] = out["top1_prob"].fillna(0.0) - out["top2_prob"].fillna(0.0)
    out["top_score_margin"] = out["top1_score"].fillna(0.0) - out["top2_score"].fillna(0.0)

    # Anchor distance from selected path anchors
    out["gate_anchor_dist_px"] = np.sqrt(
        (out["gate_anchor1_x"] - out["gate_anchor2_x"]) ** 2
        + (out["gate_anchor1_y"] - out["gate_anchor2_y"]) ** 2
    )

    # Relevant teacher anchor confidences
    anchor_min_conf = []
    for _, row in out.iterrows():
        target = row["target_name"]
        a_name, b_name = TARGET_TO_ANCHORS[target]

        ca = safe_float(row.get(f"teacher_{a_name}_conf", np.nan), np.nan)
        cb = safe_float(row.get(f"teacher_{b_name}_conf", np.nan), np.nan)

        if not np.isfinite(ca):
            ca = 1.0
        if not np.isfinite(cb):
            cb = 1.0

        anchor_min_conf.append(min(ca, cb))

    out["gate_anchor_min_conf"] = anchor_min_conf

    # Agreement between top1 path and region head
    out["top1_region_dist_px"] = [
        dist2d(a, b, c, d)
        for a, b, c, d in zip(
            out["top1_x"], out["top1_y"],
            out["teacher_region_x"], out["teacher_region_y"],
        )
    ]

    # Agreement between soft output and top1
    if "teacher_anchor_soft_x" in out.columns:
        out["top1_soft_dist_px"] = [
            dist2d(a, b, c, d)
            for a, b, c, d in zip(
                out["top1_x"], out["top1_y"],
                out["teacher_anchor_soft_x"], out["teacher_anchor_soft_y"],
            )
        ]
    else:
        out["top1_soft_dist_px"] = np.nan

    # Convenience baseline column names
    if "teacher_anchor_soft_error_px" in out.columns:
        out["soft_error_px"] = out["teacher_anchor_soft_error_px"]
    if "teacher_region_error_px" in out.columns:
        out["region_error_px_gate"] = out["teacher_region_error_px"]

    return out


def apply_gate(df, cfg):
    out = df.copy()

    use_top1 = (
        (out["gate_anchor_dist_px"] >= cfg["min_anchor_dist"])
        & (out["gate_anchor_dist_px"] <= cfg["max_anchor_dist"])
        & (out["gate_anchor_min_conf"] >= cfg["min_anchor_conf"])
        & (out["top1_prob"].fillna(0.0) >= cfg["min_top1_prob"])
        & (out["top_prob_margin"].fillna(0.0) >= cfg["min_prob_margin"])
        & (out["top1_region_dist_px"].fillna(999999.0) <= cfg["max_top1_region_dist"])
    )

    if "top1_soft_dist_px" in out.columns:
        use_top1 = use_top1 & (out["top1_soft_dist_px"].fillna(999999.0) <= cfg["max_top1_soft_dist"])

    out["gate_use_top1"] = use_top1.astype(int)
    out["gate_source"] = np.where(out["gate_use_top1"] == 1, "top1_slime_path", "region_fallback")

    out["gated_x"] = np.where(out["gate_use_top1"] == 1, out["top1_x"], out["teacher_region_x"])
    out["gated_y"] = np.where(out["gate_use_top1"] == 1, out["top1_y"], out["teacher_region_y"])

    out["gated_error_px"] = [
        dist2d(a, b, c, d)
        for a, b, c, d in zip(
            out["gated_x"], out["gated_y"],
            out["target_orig_x"], out["target_orig_y"],
        )
    ]

    if "yolo_target_error_px" in out.columns:
        out["gated_better_than_yolo"] = out["gated_error_px"] < out["yolo_target_error_px"]

    out["gated_better_than_region"] = out["gated_error_px"] < out["teacher_region_error_px"]
    out["gated_better_than_top1"] = out["gated_error_px"] < out["top1_error_px"]
    if "teacher_anchor_soft_error_px" in out.columns:
        out["gated_better_than_soft"] = out["gated_error_px"] < out["teacher_anchor_soft_error_px"]

    return out


def summarize(df):
    out = {
        "count": len(df),
        "gated_mean_px": df["gated_error_px"].mean(),
        "gated_median_px": df["gated_error_px"].median(),
        "gate_use_top1_rate": df["gate_use_top1"].mean(),

        "top1_mean_px": df["top1_error_px"].mean(),
        "top1_median_px": df["top1_error_px"].median(),

        "region_mean_px": df["teacher_region_error_px"].mean(),
        "region_median_px": df["teacher_region_error_px"].median(),
    }

    if "teacher_anchor_soft_error_px" in df.columns:
        out["soft_mean_px"] = df["teacher_anchor_soft_error_px"].mean()
        out["soft_median_px"] = df["teacher_anchor_soft_error_px"].median()

    if "top3_oracle_error_px" in df.columns:
        out["top3_oracle_mean_px"] = df["top3_oracle_error_px"].mean()
    if "top5_oracle_error_px" in df.columns:
        out["top5_oracle_mean_px"] = df["top5_oracle_error_px"].mean()

    if "yolo_target_error_px" in df.columns:
        out["yolo_mean_px"] = df["yolo_target_error_px"].mean()
        out["yolo_median_px"] = df["yolo_target_error_px"].median()
        out["gated_win_vs_yolo"] = (df["gated_error_px"] < df["yolo_target_error_px"]).mean()
        out["top1_win_vs_yolo"] = (df["top1_error_px"] < df["yolo_target_error_px"]).mean()
        out["region_win_vs_yolo"] = (df["teacher_region_error_px"] < df["yolo_target_error_px"]).mean()

    out["gated_win_vs_region"] = (df["gated_error_px"] < df["teacher_region_error_px"]).mean()
    out["gated_win_vs_top1"] = (df["gated_error_px"] < df["top1_error_px"]).mean()

    if "teacher_anchor_soft_error_px" in df.columns:
        out["gated_win_vs_soft"] = (df["gated_error_px"] < df["teacher_anchor_soft_error_px"]).mean()

    return out


def error_bins(df):
    if "yolo_target_error_px" not in df.columns:
        return pd.DataFrame()

    bins = [0, 20, 40, 60, 80, 100, 150, 999999]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"]

    tmp = df.copy()
    tmp["yolo_error_bin"] = pd.cut(
        tmp["yolo_target_error_px"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    agg = {
        "count": ("gated_error_px", "count"),
        "yolo_mean_px": ("yolo_target_error_px", "mean"),
        "gated_mean_px": ("gated_error_px", "mean"),
        "top1_mean_px": ("top1_error_px", "mean"),
        "region_mean_px": ("teacher_region_error_px", "mean"),
        "gate_use_top1_rate": ("gate_use_top1", "mean"),
        "gated_win_vs_yolo": ("gated_better_than_yolo", "mean"),
    }

    if "teacher_anchor_soft_error_px" in tmp.columns:
        agg["soft_mean_px"] = ("teacher_anchor_soft_error_px", "mean")
    if "top5_oracle_error_px" in tmp.columns:
        agg["top5_oracle_mean_px"] = ("top5_oracle_error_px", "mean")

    return tmp.groupby("yolo_error_bin", observed=False).agg(**agg).reset_index()
