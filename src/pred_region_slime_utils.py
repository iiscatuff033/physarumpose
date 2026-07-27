import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .pred_region_common import (
    TARGET_TO_ANCHORS,
    TARGET_TO_ID,
    get_target_anchors,
    valid_point,
    safe_float,
    point_error,
    save_json,
    load_json,
    split_by_image,
)


def make_edge_map(image_bgr, blur=3, low=50, high=150):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(gray, low, high)
    return edges.astype(np.float32) / 255.0


def sample_line_points(p1, p2, n=80):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    ts = np.linspace(0.0, 1.0, n)
    return p1[None, :] * (1.0 - ts[:, None]) + p2[None, :] * ts[:, None]


def edge_support_segment(edge_map, p1, p2, n=80):
    h, w = edge_map.shape[:2]
    pts = sample_line_points(p1, p2, n=n)
    vals = []
    for x, y in pts:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        xi = int(round(x)); yi = int(round(y))
        xi = max(0, min(w - 1, xi)); yi = max(0, min(h - 1, yi))
        vals.append(edge_map[yi, xi])
    return float(np.mean(vals)) if vals else 0.0


def local_edge_score(edge_map, c, radius=5):
    h, w = edge_map.shape[:2]
    x, y = int(round(float(c[0]))), int(round(float(c[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0.0
    x1 = max(0, x - radius); x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius); y2 = min(h, y + radius + 1)
    patch = edge_map[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def angle_at_candidate(a, c, b):
    v1 = a - c; v2 = b - c
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def anatomy_score(a, c, b):
    angle = angle_at_candidate(a, c, b)
    if angle < 25:
        angle_score = angle / 25.0
    elif angle > 178:
        angle_score = max(0.3, 1.0 - (angle - 178.0) / 10.0)
    else:
        angle_score = 1.0
    d1 = float(np.linalg.norm(a - c)); d2 = float(np.linalg.norm(c - b)); direct = float(np.linalg.norm(a - b)) + 1e-6
    min_ratio = min(d1, d2) / max(d1 + d2, 1e-6)
    balance_score = float(np.clip(min_ratio / 0.25, 0.0, 1.0))
    path_ratio = (d1 + d2) / direct
    length_score = 1.0 if path_ratio <= 1.8 else max(0.0, 1.0 - (path_ratio - 1.8) / 1.2)
    return float(0.45 * angle_score + 0.35 * balance_score + 0.20 * length_score)


def line_coordinates(a, b, c):
    v = b - a
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return 0.0, 0.0
    unit = v / d
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    rel = c - a
    alpha = float(np.dot(rel, unit) / d)
    beta = float(np.dot(rel, perp) / d)
    return alpha, beta


def generate_candidates(a, b, image_shape, pred_region=None, grid_alphas=None, grid_betas=None):
    h, w = image_shape[:2]
    if grid_alphas is None:
        grid_alphas = np.linspace(0.18, 0.82, 17)
    if grid_betas is None:
        grid_betas = np.linspace(-0.95, 0.95, 25)
    a = np.array(a, dtype=np.float32); b = np.array(b, dtype=np.float32)
    v = b - a; d = float(np.linalg.norm(v))
    if d < 5:
        return np.zeros((0, 2), dtype=np.float32)
    unit = v / d
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    cand = []
    for alpha in grid_alphas:
        base = a + alpha * v
        for beta in grid_betas:
            p = base + beta * d * perp
            x, y = float(p[0]), float(p[1])
            if 0 <= x < w and 0 <= y < h:
                cand.append([x, y])

    # Add additional candidates around predicted region, not GT occlusion box.
    if pred_region is not None and np.isfinite(pred_region).all() and pred_region[0] > 0 and pred_region[1] > 0:
        px, py = float(pred_region[0]), float(pred_region[1])
        radius = max(0.25 * d, 15.0)
        for rx in [-0.7, -0.35, 0.0, 0.35, 0.7]:
            for ry in [-0.7, -0.35, 0.0, 0.35, 0.7]:
                x = px + rx * radius
                y = py + ry * radius
                if 0 <= x < w and 0 <= y < h:
                    cand.append([x, y])
    if not cand:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.array(cand, dtype=np.float32)
    rounded = np.round(arr / 3.0).astype(np.int32)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return arr[np.sort(unique_idx)]


def slime_reinforce(candidates, raw_scores, anchor_dist, iterations=8, diffusion=0.20, decay=0.08):
    if len(candidates) == 0:
        return raw_scores
    flow = raw_scores.astype(np.float32).copy()
    pts = candidates.astype(np.float32)
    sigma = max(0.20 * anchor_dist, 8.0)
    for _ in range(iterations):
        new_flow = (1.0 - decay) * flow + raw_scores
        diffused = np.zeros_like(flow)
        for i in range(len(pts)):
            d2 = np.sum((pts - pts[i]) ** 2, axis=1)
            weights = np.exp(-d2 / (2.0 * sigma * sigma))
            weights = weights / (weights.sum() + 1e-6)
            diffused[i] = float(np.sum(weights * flow))
        flow = (1.0 - diffusion) * new_flow + diffusion * diffused
        maxv = float(flow.max())
        if maxv > 0:
            flow = flow / maxv
    return flow


PRED_REGION_FEATURES = [
    "anchor1_conf", "anchor2_conf", "anchor_min_conf",
    "edge1", "edge2", "edge_mean", "edge_min", "local_edge", "anatomy_score",
    "angle_deg", "alpha", "beta", "abs_beta",
    "d1_norm", "d2_norm", "direct_norm", "path_ratio", "balance",
    "candidate_x_norm", "candidate_y_norm", "anchor1_x_norm", "anchor1_y_norm", "anchor2_x_norm", "anchor2_y_norm",
    "heuristic_raw", "heuristic_slime_score",
    "pred_region_x_norm", "pred_region_y_norm", "pred_region_conf",
    "dist_to_pred_region_norm", "pred_region_score", "near_pred_region",
    "target_id_0", "target_id_1", "target_id_2", "target_id_3",
]


def build_candidate_features(row, image_bgr, candidates, a, b, ac, bc):
    edge_map = make_edge_map(image_bgr)
    h, w = image_bgr.shape[:2]
    target_name = row["target_name"]
    target_id = TARGET_TO_ID[target_name]
    anchor_dist = float(np.linalg.norm(a - b))
    anchor_dist = max(anchor_dist, 1.0)

    prx = safe_float(row.get("region_pred_x", np.nan))
    pry = safe_float(row.get("region_pred_y", np.nan))
    prconf = safe_float(row.get("region_pred_conf", 0.0), 0.0)
    pred_region = np.array([prx, pry], dtype=np.float32)

    records = []
    raw_scores = []
    for i, c in enumerate(candidates):
        c = np.array(c, dtype=np.float32)
        d1 = float(np.linalg.norm(a - c)); d2 = float(np.linalg.norm(c - b)); direct = float(np.linalg.norm(a - b)); path_len = d1 + d2
        angle = angle_at_candidate(a, c, b)
        alpha, beta = line_coordinates(a, b, c)
        edge1 = edge_support_segment(edge_map, a, c)
        edge2 = edge_support_segment(edge_map, c, b)
        edge_mean = float((edge1 + edge2) / 2.0); edge_min = float(min(edge1, edge2))
        local_edge = local_edge_score(edge_map, c)
        anat = anatomy_score(a, c, b)
        balance = float(min(d1, d2) / max(path_len, 1e-6))
        path_ratio = float(path_len / max(direct, 1.0))
        if np.isfinite(prx) and np.isfinite(pry) and prx > 0 and pry > 0:
            dist_pr = float(np.sqrt((c[0] - prx) ** 2 + (c[1] - pry) ** 2))
            dist_pr_norm = dist_pr / anchor_dist
            pred_region_score = float(np.exp(-dist_pr / max(0.25 * anchor_dist, 8.0))) * float(prconf)
            near_pred_region = 1.0 if dist_pr_norm < 0.35 else 0.0
        else:
            dist_pr_norm = 9.9
            pred_region_score = 0.0
            near_pred_region = 0.0
        heuristic_raw = (
            0.75 * edge_mean + 0.50 * edge_min + 1.10 * anat + 0.20 * local_edge +
            1.40 * pred_region_score + 0.15 * min(ac, bc)
        )
        raw_scores.append(heuristic_raw)
        tx = safe_float(row.get("target_orig_x", np.nan)); ty = safe_float(row.get("target_orig_y", np.nan))
        cand_err = point_error(c[0], c[1], tx, ty)
        sigma = max(0.12 * anchor_dist, 8.0)
        quality = float(np.exp(-cand_err / sigma)) if np.isfinite(cand_err) else 0.0
        rec = {
            "sample_id": int(row["sample_id"]), "candidate_id": int(i),
            "image_name": row.get("image_name", ""), "occluded_image_name": row.get("occluded_image_name", ""), "target_name": target_name,
            "candidate_x": float(c[0]), "candidate_y": float(c[1]), "target_orig_x": tx, "target_orig_y": ty,
            "candidate_error_px": cand_err, "candidate_quality": quality,
            "anchor1_x": float(a[0]), "anchor1_y": float(a[1]), "anchor2_x": float(b[0]), "anchor2_y": float(b[1]),
            "anchor1_conf": float(ac), "anchor2_conf": float(bc), "anchor_min_conf": float(min(ac, bc)),
            "edge1": edge1, "edge2": edge2, "edge_mean": edge_mean, "edge_min": edge_min, "local_edge": local_edge, "anatomy_score": anat,
            "angle_deg": angle, "alpha": alpha, "beta": beta, "abs_beta": abs(beta),
            "d1_norm": d1 / anchor_dist, "d2_norm": d2 / anchor_dist, "direct_norm": direct / anchor_dist,
            "path_ratio": path_ratio, "balance": balance,
            "candidate_x_norm": float(c[0]) / max(float(w), 1.0), "candidate_y_norm": float(c[1]) / max(float(h), 1.0),
            "anchor1_x_norm": float(a[0]) / max(float(w), 1.0), "anchor1_y_norm": float(a[1]) / max(float(h), 1.0),
            "anchor2_x_norm": float(b[0]) / max(float(w), 1.0), "anchor2_y_norm": float(b[1]) / max(float(h), 1.0),
            "heuristic_raw": heuristic_raw,
            "pred_region_x": prx, "pred_region_y": pry, "pred_region_conf": prconf,
            "pred_region_x_norm": prx / max(float(w), 1.0) if np.isfinite(prx) else 0.0,
            "pred_region_y_norm": pry / max(float(h), 1.0) if np.isfinite(pry) else 0.0,
            "dist_to_pred_region_norm": dist_pr_norm, "pred_region_score": pred_region_score, "near_pred_region": near_pred_region,
            "target_id_0": 1.0 if target_id == 0 else 0.0, "target_id_1": 1.0 if target_id == 1 else 0.0,
            "target_id_2": 1.0 if target_id == 2 else 0.0, "target_id_3": 1.0 if target_id == 3 else 0.0,
        }
        for col in ["yolo_target_x", "yolo_target_y", "yolo_target_error_px", "prior_pred_x", "prior_pred_y", "prior_error_px", "geometry_x", "geometry_y", "geometry_error_px", "region_error_px"]:
            if col in row.index:
                rec[col] = safe_float(row.get(col, np.nan))
        records.append(rec)
    if not records:
        return pd.DataFrame()
    raw_scores = np.array(raw_scores, dtype=np.float32)
    slime_scores = slime_reinforce(np.array([[r["candidate_x"], r["candidate_y"]] for r in records], dtype=np.float32), raw_scores, anchor_dist)
    for r, s in zip(records, slime_scores):
        r["heuristic_slime_score"] = float(s)
    return pd.DataFrame(records)


def summarize_best(best_df, method_col="learned_error_px"):
    out = {
        "count": len(best_df),
        "learned_mean_px": best_df[method_col].mean(),
        "learned_median_px": best_df[method_col].median(),
        "heuristic_mean_px": best_df["heuristic_error_px"].mean() if "heuristic_error_px" in best_df.columns else np.nan,
        "oracle_mean_px": best_df["oracle_error_px"].mean() if "oracle_error_px" in best_df.columns else np.nan,
        "region_mean_px": best_df["region_error_px"].mean() if "region_error_px" in best_df.columns else np.nan,
    }
    if "yolo_target_error_px" in best_df.columns:
        out["yolo_mean_px"] = best_df["yolo_target_error_px"].mean()
        out["learned_win_vs_yolo"] = (best_df[method_col] < best_df["yolo_target_error_px"]).mean()
    return out
