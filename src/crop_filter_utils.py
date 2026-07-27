import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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


def get_image_size(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return w, h


def crop_box_from_points(points, img_w, img_h, pad=120, min_size=260):
    xs = []
    ys = []

    for p in points:
        if p is None:
            continue
        x, y = p
        x = safe_float(x)
        y = safe_float(y)
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)

    if len(xs) == 0:
        return 0, 0, img_w, img_h

    x1 = max(0, int(min(xs) - pad))
    y1 = max(0, int(min(ys) - pad))
    x2 = min(img_w, int(max(xs) + pad))
    y2 = min(img_h, int(max(ys) + pad))

    # Enforce minimum size around current center.
    if (x2 - x1) < min_size:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - min_size // 2)
        x2 = min(img_w, x1 + min_size)
        x1 = max(0, x2 - min_size)

    if (y2 - y1) < min_size:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - min_size // 2)
        y2 = min(img_h, y1 + min_size)
        y1 = max(0, y2 - min_size)

    return int(x1), int(y1), int(x2), int(y2)


def point_inside_crop(x, y, x1, y1, x2, y2):
    x = safe_float(x)
    y = safe_float(y)
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    return x1 <= x <= x2 and y1 <= y <= y2


def merge_topk_if_needed(gated_df, topk_df):
    out = gated_df.copy()

    # Make sure top1 columns exist.
    if "top1_x" not in out.columns or "top1_y" not in out.columns:
        r1 = topk_df[topk_df["rank"] == 1].copy()
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
        keep = [
            "sample_id", "top1_x", "top1_y", "top1_error_px", "top1_prob", "top1_score",
            "gate_anchor1_x", "gate_anchor1_y", "gate_anchor2_x", "gate_anchor2_y",
        ]
        keep = [c for c in keep if c in r1.columns]
        out = out.merge(r1[keep], on="sample_id", how="left")

    return out


def add_crop_quality_flags(df):
    out = df.copy()

    # Core diagnostic distances
    out["anchor_dist_px"] = [
        dist2d(a, b, c, d)
        for a, b, c, d in zip(
            out["gate_anchor1_x"], out["gate_anchor1_y"],
            out["gate_anchor2_x"], out["gate_anchor2_y"],
        )
    ]

    out["top1_region_dist_px"] = [
        dist2d(a, b, c, d)
        for a, b, c, d in zip(
            out["top1_x"], out["top1_y"],
            out["teacher_region_x"], out["teacher_region_y"],
        )
    ]

    out["gated_region_dist_px"] = [
        dist2d(a, b, c, d)
        for a, b, c, d in zip(
            out["gated_x"], out["gated_y"],
            out["teacher_region_x"], out["teacher_region_y"],
        )
    ]

    # Suspicious rules. These are visual/inference quality flags, not training labels.
    out["flag_anchor_too_short"] = out["anchor_dist_px"] < 15
    out["flag_anchor_too_long"] = out["anchor_dist_px"] > 300
    out["flag_top1_far_from_region"] = out["top1_region_dist_px"] > 100
    out["flag_gated_far_from_region"] = out["gated_region_dist_px"] > 120
    out["flag_large_error"] = out["gated_error_px"] > 40 if "gated_error_px" in out.columns else False

    out["suspicious_case"] = (
        out["flag_anchor_too_short"]
        | out["flag_anchor_too_long"]
        | out["flag_top1_far_from_region"]
        | out["flag_gated_far_from_region"]
    )

    if "gated_error_px" in out.columns:
        out["good_paper_example"] = (
            (out["gated_error_px"] <= 20)
            & (~out["suspicious_case"])
            & (out["anchor_dist_px"] >= 20)
            & (out["anchor_dist_px"] <= 220)
        )
    else:
        out["good_paper_example"] = ~out["suspicious_case"]

    return out


def summarize_crop_quality(df):
    out = {
        "count": len(df),
        "suspicious_rate": float(df["suspicious_case"].mean()) if len(df) else np.nan,
        "good_paper_example_rate": float(df["good_paper_example"].mean()) if len(df) else np.nan,
        "mean_anchor_dist_px": float(df["anchor_dist_px"].mean()) if "anchor_dist_px" in df.columns else np.nan,
        "median_anchor_dist_px": float(df["anchor_dist_px"].median()) if "anchor_dist_px" in df.columns else np.nan,
        "mean_top1_region_dist_px": float(df["top1_region_dist_px"].mean()) if "top1_region_dist_px" in df.columns else np.nan,
        "median_top1_region_dist_px": float(df["top1_region_dist_px"].median()) if "top1_region_dist_px" in df.columns else np.nan,
    }

    if "gated_error_px" in df.columns:
        out["gated_mean_px_all"] = float(df["gated_error_px"].mean())
        out["gated_median_px_all"] = float(df["gated_error_px"].median())

        good = df[df["good_paper_example"]]
        susp = df[df["suspicious_case"]]
        if len(good):
            out["gated_mean_px_good_examples"] = float(good["gated_error_px"].mean())
            out["gated_median_px_good_examples"] = float(good["gated_error_px"].median())
        else:
            out["gated_mean_px_good_examples"] = np.nan
            out["gated_median_px_good_examples"] = np.nan

        if len(susp):
            out["gated_mean_px_suspicious"] = float(susp["gated_error_px"].mean())
            out["gated_median_px_suspicious"] = float(susp["gated_error_px"].median())
        else:
            out["gated_mean_px_suspicious"] = np.nan
            out["gated_median_px_suspicious"] = np.nan

    return out


def shift_points_in_row(row, dx, dy):
    row = row.copy()
    for col in list(row.index):
        if col.endswith("_x"):
            try:
                row[col] = row[col] - dx
            except Exception:
                pass
        elif col.endswith("_y"):
            try:
                row[col] = row[col] - dy
            except Exception:
                pass
    return row
