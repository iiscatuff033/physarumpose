import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TARGET_TO_ANCHORS = {
    "right_elbow": ("right_shoulder", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_wrist"),
    "right_knee": ("right_hip", "right_ankle"),
    "left_knee": ("left_hip", "left_ankle"),
}

TARGET_TO_ID = {
    "right_elbow": 0,
    "left_elbow": 1,
    "right_knee": 2,
    "left_knee": 3,
}

ANCHOR_ORDER = [
    "right_shoulder",
    "right_wrist",
    "right_hip",
    "right_ankle",
    "left_shoulder",
    "left_wrist",
    "left_hip",
    "left_ankle",
]

ANCHOR_INDEX = {name: i for i, name in enumerate(ANCHOR_ORDER)}


COARSE_REGION_FEATURES = [
    "anchor1_conf",
    "anchor2_conf",
    "anchor_min_conf",

    "edge1",
    "edge2",
    "edge_mean",
    "edge_min",
    "local_edge",
    "anatomy_score",

    "angle_deg",
    "alpha",
    "beta",
    "abs_beta",

    "d1_norm",
    "d2_norm",
    "direct_norm",
    "path_ratio",
    "balance",

    "candidate_x_norm",
    "candidate_y_norm",
    "anchor1_x_norm",
    "anchor1_y_norm",
    "anchor2_x_norm",
    "anchor2_y_norm",

    "heuristic_raw",
    "heuristic_slime_score",

    "coarse_region_x_norm",
    "coarse_region_y_norm",
    "coarse_region_conf",
    "coarse_cell_px_norm",
    "dist_to_coarse_region_norm",
    "coarse_region_score",
    "near_coarse_region",

    "target_id_0",
    "target_id_1",
    "target_id_2",
    "target_id_3",
]


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


def point_error(px, py, tx, ty):
    if np.isnan(px) or np.isnan(py):
        return np.nan
    return float(np.sqrt((px - tx) ** 2 + (py - ty) ** 2))


def find_xy_conf_columns(row, anchor_name):
    idx = ANCHOR_INDEX[anchor_name]
    patterns = [
        (f"{anchor_name}_x", f"{anchor_name}_y", f"{anchor_name}_conf"),
        (f"{anchor_name}_pred_x", f"{anchor_name}_pred_y", f"{anchor_name}_conf"),
        (f"pred_{anchor_name}_x", f"pred_{anchor_name}_y", f"pred_{anchor_name}_conf"),
        (f"anchor_{anchor_name}_x", f"anchor_{anchor_name}_y", f"anchor_{anchor_name}_conf"),
        (f"anchornet_{anchor_name}_x", f"anchornet_{anchor_name}_y", f"anchornet_{anchor_name}_conf"),
        (f"strong_anchor_{anchor_name}_x", f"strong_anchor_{anchor_name}_y", f"strong_anchor_{anchor_name}_conf"),
        (f"strong_{anchor_name}_x", f"strong_{anchor_name}_y", f"strong_{anchor_name}_conf"),
        (f"anchor{idx}_x", f"anchor{idx}_y", f"anchor{idx}_conf"),
        (f"pred_anchor{idx}_x", f"pred_anchor{idx}_y", f"pred_anchor{idx}_conf"),
        (f"a{idx}_x", f"a{idx}_y", f"a{idx}_conf"),
        (f"pred_a{idx}_x", f"pred_a{idx}_y", f"pred_a{idx}_conf"),
    ]

    for xk, yk, ck in patterns:
        if xk in row.index and yk in row.index:
            conf = row[ck] if ck in row.index else 1.0
            return xk, yk, ck if ck in row.index else None, safe_float(row[xk]), safe_float(row[yk]), safe_float(conf, 1.0)

    lower_cols = {c.lower(): c for c in row.index}
    name = anchor_name.lower()
    xs, ys, cs = [], [], []
    for lc, original in lower_cols.items():
        if name in lc and lc.endswith("_x"):
            xs.append(original)
        if name in lc and lc.endswith("_y"):
            ys.append(original)
        if name in lc and ("conf" in lc or "score" in lc):
            cs.append(original)

    if xs and ys:
        xk = xs[0]
        yk = ys[0]
        ck = cs[0] if cs else None
        conf = row[ck] if ck else 1.0
        return xk, yk, ck, safe_float(row[xk]), safe_float(row[yk]), safe_float(conf, 1.0)

    return None, None, None, np.nan, np.nan, 0.0


def get_anchor(row, anchor_name):
    _, _, _, x, y, conf = find_xy_conf_columns(row, anchor_name)
    return np.array([x, y], dtype=np.float32), float(conf)


def get_target_anchors(row):
    target = row["target_name"]
    a_name, b_name = TARGET_TO_ANCHORS[target]
    a, ac = get_anchor(row, a_name)
    b, bc = get_anchor(row, b_name)
    return a_name, b_name, a, b, ac, bc


def valid_point(p):
    return np.isfinite(p).all() and p[0] > 0 and p[1] > 0


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
        xi = int(round(x))
        yi = int(round(y))
        xi = max(0, min(w - 1, xi))
        yi = max(0, min(h - 1, yi))
        vals.append(edge_map[yi, xi])
    return float(np.mean(vals)) if vals else 0.0


def local_edge_score(edge_map, c, radius=5):
    h, w = edge_map.shape[:2]
    x = int(round(float(c[0])))
    y = int(round(float(c[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0.0
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    patch = edge_map[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def angle_at_candidate(a, c, b):
    v1 = a - c
    v2 = b - c
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
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

    d1 = float(np.linalg.norm(a - c))
    d2 = float(np.linalg.norm(c - b))
    direct = float(np.linalg.norm(a - b)) + 1e-6

    min_ratio = min(d1, d2) / max(d1 + d2, 1e-6)
    balance_score = float(np.clip(min_ratio / 0.25, 0.0, 1.0))

    path_ratio = (d1 + d2) / direct
    if path_ratio <= 1.8:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - (path_ratio - 1.8) / 1.2)

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


def quantize_pred_region(row, image_shape, cell_px=64, jitter_px=0, seed=42):
    """
    Convert exact region_pred_x/y into a coarse region center.

    The exact region point is NOT used as a model feature.
    It is only used here to form a coarse grid cell.
    """
    h, w = image_shape[:2]
    rx = safe_float(row.get("region_pred_x", np.nan))
    ry = safe_float(row.get("region_pred_y", np.nan))
    conf = safe_float(row.get("region_pred_conf", 0.0), 0.0)

    if not np.isfinite(rx) or not np.isfinite(ry) or rx <= 0 or ry <= 0:
        return np.array([np.nan, np.nan], dtype=np.float32), 0.0

    cx = (math.floor(rx / cell_px) * cell_px) + (cell_px / 2.0)
    cy = (math.floor(ry / cell_px) * cell_px) + (cell_px / 2.0)

    if jitter_px > 0:
        sample_id = int(row.get("sample_id", 0))
        rng = np.random.default_rng(seed + sample_id)
        cx += float(rng.normal(0.0, jitter_px))
        cy += float(rng.normal(0.0, jitter_px))

    cx = float(np.clip(cx, 0, max(w - 1, 1)))
    cy = float(np.clip(cy, 0, max(h - 1, 1)))

    return np.array([cx, cy], dtype=np.float32), float(conf)


def generate_candidates_coarse(a, b, image_shape, coarse_region=None, cell_px=64, grid_alphas=None, grid_betas=None):
    h, w = image_shape[:2]

    if grid_alphas is None:
        grid_alphas = np.linspace(0.16, 0.84, 19)

    if grid_betas is None:
        grid_betas = np.linspace(-1.05, 1.05, 31)

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    v = b - a
    d = float(np.linalg.norm(v))

    if d < 5:
        return np.zeros((0, 2), dtype=np.float32)

    unit = v / d
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)

    cand = []

    # Global anchor-to-anchor candidate fan.
    for alpha in grid_alphas:
        base = a + alpha * v
        for beta in grid_betas:
            p = base + beta * d * perp
            x, y = float(p[0]), float(p[1])
            if 0 <= x < w and 0 <= y < h:
                cand.append([x, y])

    # Coarse cell candidate fan.
    # This is intentionally broad. It is not an exact point.
    if coarse_region is not None and np.isfinite(coarse_region).all() and coarse_region[0] > 0 and coarse_region[1] > 0:
        cx, cy = float(coarse_region[0]), float(coarse_region[1])
        radius = max(float(cell_px) * 1.25, 0.22 * d, 20.0)
        steps = [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0]
        for sx in steps:
            for sy in steps:
                x = cx + sx * radius
                y = cy + sy * radius
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


def build_candidate_features(row, image_bgr, candidates, a, b, ac, bc, coarse_region, coarse_conf, cell_px):
    edge_map = make_edge_map(image_bgr)
    h, w = image_bgr.shape[:2]

    target_name = row["target_name"]
    target_id = TARGET_TO_ID[target_name]

    anchor_dist = float(np.linalg.norm(a - b))
    anchor_dist = max(anchor_dist, 1.0)

    cx = safe_float(coarse_region[0])
    cy = safe_float(coarse_region[1])

    records = []
    raw_scores = []

    for i, c in enumerate(candidates):
        c = np.array(c, dtype=np.float32)

        d1 = float(np.linalg.norm(a - c))
        d2 = float(np.linalg.norm(c - b))
        direct = float(np.linalg.norm(a - b))
        path_len = d1 + d2

        angle = angle_at_candidate(a, c, b)
        alpha, beta = line_coordinates(a, b, c)

        edge1 = edge_support_segment(edge_map, a, c)
        edge2 = edge_support_segment(edge_map, c, b)
        edge_mean = float((edge1 + edge2) / 2.0)
        edge_min = float(min(edge1, edge2))
        local_edge = local_edge_score(edge_map, c)

        anat = anatomy_score(a, c, b)
        balance = float(min(d1, d2) / max(path_len, 1e-6))
        path_ratio = float(path_len / max(direct, 1.0))

        if np.isfinite(cx) and np.isfinite(cy):
            dist_cr = float(np.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2))
            dist_cr_norm = dist_cr / anchor_dist

            # Broad region score: deliberately less precise than Stage 5C.
            sigma = max(float(cell_px), 0.28 * anchor_dist, 18.0)
            coarse_region_score = float(np.exp(-dist_cr / sigma)) * float(coarse_conf)
            near_coarse_region = 1.0 if dist_cr <= 0.90 * float(cell_px) else 0.0
        else:
            dist_cr_norm = 9.9
            coarse_region_score = 0.0
            near_coarse_region = 0.0

        heuristic_raw = (
            0.70 * edge_mean
            + 0.45 * edge_min
            + 1.15 * anat
            + 0.20 * local_edge
            + 0.90 * coarse_region_score
            + 0.15 * min(ac, bc)
        )
        raw_scores.append(heuristic_raw)

        tx = safe_float(row.get("target_orig_x", np.nan))
        ty = safe_float(row.get("target_orig_y", np.nan))
        cand_err = point_error(c[0], c[1], tx, ty)

        label_sigma = max(0.12 * anchor_dist, 8.0)
        quality = float(np.exp(-cand_err / label_sigma)) if np.isfinite(cand_err) else 0.0

        rec = {
            "sample_id": int(row["sample_id"]),
            "candidate_id": int(i),
            "image_name": row.get("image_name", ""),
            "occluded_image_name": row.get("occluded_image_name", ""),
            "target_name": target_name,

            "candidate_x": float(c[0]),
            "candidate_y": float(c[1]),
            "target_orig_x": tx,
            "target_orig_y": ty,
            "candidate_error_px": cand_err,
            "candidate_quality": quality,

            "anchor1_x": float(a[0]),
            "anchor1_y": float(a[1]),
            "anchor2_x": float(b[0]),
            "anchor2_y": float(b[1]),
            "anchor1_conf": float(ac),
            "anchor2_conf": float(bc),
            "anchor_min_conf": float(min(ac, bc)),

            "edge1": edge1,
            "edge2": edge2,
            "edge_mean": edge_mean,
            "edge_min": edge_min,
            "local_edge": local_edge,
            "anatomy_score": anat,

            "angle_deg": angle,
            "alpha": alpha,
            "beta": beta,
            "abs_beta": abs(beta),

            "d1_norm": d1 / anchor_dist,
            "d2_norm": d2 / anchor_dist,
            "direct_norm": direct / anchor_dist,
            "path_ratio": path_ratio,
            "balance": balance,

            "candidate_x_norm": float(c[0]) / max(float(w), 1.0),
            "candidate_y_norm": float(c[1]) / max(float(h), 1.0),
            "anchor1_x_norm": float(a[0]) / max(float(w), 1.0),
            "anchor1_y_norm": float(a[1]) / max(float(h), 1.0),
            "anchor2_x_norm": float(b[0]) / max(float(w), 1.0),
            "anchor2_y_norm": float(b[1]) / max(float(h), 1.0),

            "heuristic_raw": heuristic_raw,

            "coarse_region_x": cx,
            "coarse_region_y": cy,
            "coarse_region_conf": float(coarse_conf),
            "coarse_cell_px": float(cell_px),
            "coarse_region_x_norm": cx / max(float(w), 1.0) if np.isfinite(cx) else 0.0,
            "coarse_region_y_norm": cy / max(float(h), 1.0) if np.isfinite(cy) else 0.0,
            "coarse_cell_px_norm": float(cell_px) / max(float(max(w, h)), 1.0),
            "dist_to_coarse_region_norm": dist_cr_norm,
            "coarse_region_score": coarse_region_score,
            "near_coarse_region": near_coarse_region,

            # Exact region values are recorded for evaluation only.
            # They are NOT included in COARSE_REGION_FEATURES.
            "region_pred_x": safe_float(row.get("region_pred_x", np.nan)),
            "region_pred_y": safe_float(row.get("region_pred_y", np.nan)),
            "region_pred_conf": safe_float(row.get("region_pred_conf", np.nan)),
            "region_error_px": safe_float(row.get("region_error_px", np.nan)),

            "target_id_0": 1.0 if target_id == 0 else 0.0,
            "target_id_1": 1.0 if target_id == 1 else 0.0,
            "target_id_2": 1.0 if target_id == 2 else 0.0,
            "target_id_3": 1.0 if target_id == 3 else 0.0,
        }

        # Baselines for comparison only, not features.
        for col in [
            "yolo_target_x", "yolo_target_y", "yolo_target_error_px",
            "prior_pred_x", "prior_pred_y", "prior_error_px",
            "geometry_x", "geometry_y", "geometry_error_px",
        ]:
            if col in row.index:
                rec[col] = safe_float(row.get(col, np.nan))

        records.append(rec)

    if not records:
        return pd.DataFrame()

    raw_scores = np.array(raw_scores, dtype=np.float32)
    slime_scores = slime_reinforce(
        np.array([[r["candidate_x"], r["candidate_y"]] for r in records], dtype=np.float32),
        raw_scores,
        anchor_dist,
    )

    for r, s in zip(records, slime_scores):
        r["heuristic_slime_score"] = float(s)

    return pd.DataFrame(records)


def split_by_image(df, val_ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)

    if "image_name" in df.columns:
        names = df["image_name"].fillna(df["occluded_image_name"]).unique()
        key = df["image_name"].fillna(df["occluded_image_name"])
    else:
        names = df["occluded_image_name"].unique()
        key = df["occluded_image_name"]

    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])

    val = df[key.isin(val_names)].copy()
    train = df[~key.isin(val_names)].copy()

    return train.reset_index(drop=True), val.reset_index(drop=True)


def summarize_best(best_df, method_col="learned_error_px"):
    out = {
        "count": len(best_df),
        "learned_mean_px": best_df[method_col].mean(),
        "learned_median_px": best_df[method_col].median(),

        "coarse_region_mean_px": best_df["coarse_region_error_px"].mean(),
        "coarse_region_median_px": best_df["coarse_region_error_px"].median(),

        "nearest_to_coarse_mean_px": best_df["nearest_coarse_error_px"].mean(),
        "nearest_to_coarse_median_px": best_df["nearest_coarse_error_px"].median(),

        "heuristic_mean_px": best_df["heuristic_error_px"].mean(),
        "heuristic_median_px": best_df["heuristic_error_px"].median(),

        "oracle_mean_px": best_df["oracle_error_px"].mean(),
        "oracle_median_px": best_df["oracle_error_px"].median(),

        "region_direct_mean_px": best_df["region_error_px"].mean() if "region_error_px" in best_df.columns else np.nan,
        "region_direct_median_px": best_df["region_error_px"].median() if "region_error_px" in best_df.columns else np.nan,

        "learned_win_vs_coarse_region": (best_df[method_col] < best_df["coarse_region_error_px"]).mean(),
        "learned_win_vs_nearest_coarse": (best_df[method_col] < best_df["nearest_coarse_error_px"]).mean(),
        "learned_win_vs_heuristic": (best_df[method_col] < best_df["heuristic_error_px"]).mean(),
    }

    if "yolo_target_error_px" in best_df.columns:
        out["yolo_mean_px"] = best_df["yolo_target_error_px"].mean()
        out["yolo_median_px"] = best_df["yolo_target_error_px"].median()
        out["learned_win_vs_yolo"] = (best_df[method_col] < best_df["yolo_target_error_px"]).mean()

    return out
