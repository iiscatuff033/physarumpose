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


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def one_hot(index, size):
    arr = np.zeros(size, dtype=np.float32)
    arr[index] = 1.0
    return arr


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


def normalize01(v, denom):
    return float(v) / max(float(denom), 1.0)


def estimate_canvas(row):
    xs, ys = [], []

    for k in row.index:
        if k.endswith("_x"):
            v = safe_float(row.get(k, np.nan))
            if np.isfinite(v) and v > 0:
                xs.append(v)
        elif k.endswith("_y"):
            v = safe_float(row.get(k, np.nan))
            if np.isfinite(v) and v > 0:
                ys.append(v)

    width = max(xs) + 50 if xs else 640.0
    height = max(ys) + 50 if ys else 480.0

    return float(width), float(height)


def find_xy_conf_columns(row, anchor_name):
    """
    Tries many possible column naming styles because earlier stages may name AnchorNet
    predictions slightly differently.
    """

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

    # Last-resort fuzzy search
    lower_cols = {c.lower(): c for c in row.index}
    name = anchor_name.lower()

    possible_x = []
    possible_y = []
    possible_c = []

    for lc, original in lower_cols.items():
        if name in lc and lc.endswith("_x"):
            possible_x.append(original)
        if name in lc and lc.endswith("_y"):
            possible_y.append(original)
        if name in lc and ("conf" in lc or "score" in lc):
            possible_c.append(original)

    if possible_x and possible_y:
        xk = possible_x[0]
        yk = possible_y[0]
        ck = possible_c[0] if possible_c else None
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

    if len(vals) == 0:
        return 0.0

    return float(np.mean(vals))


def local_edge_score(edge_map, c, radius=5):
    h, w = edge_map.shape[:2]
    x, y = int(round(float(c[0]))), int(round(float(c[1])))

    if x < 0 or y < 0 or x >= w or y >= h:
        return 0.0

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    patch = edge_map[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    return float(patch.mean())


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
    """
    Return alpha and beta for candidate c in coordinate system:
    c = a + alpha*(b-a) + beta*|b-a|*perp
    """
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


def generate_candidates_no_occ(a, b, image_shape, grid_alphas=None, grid_betas=None):
    """
    Generate candidates WITHOUT using the occlusion box.
    """
    h, w = image_shape[:2]

    if grid_alphas is None:
        grid_alphas = np.linspace(0.18, 0.82, 17)

    if grid_betas is None:
        grid_betas = np.linspace(-0.95, 0.95, 25)

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    v = b - a
    d = float(np.linalg.norm(v))

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

    if not cand:
        return np.zeros((0, 2), dtype=np.float32)

    arr = np.array(cand, dtype=np.float32)

    # Remove near duplicates
    rounded = np.round(arr / 3.0).astype(np.int32)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    arr = arr[np.sort(unique_idx)]

    return arr


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


def build_candidate_features(row, image_bgr, candidates, a, b, ac, bc):
    edge_map = make_edge_map(image_bgr)
    h, w = image_bgr.shape[:2]

    target_name = row["target_name"]
    target_id = TARGET_TO_ID[target_name]

    anchor_dist = float(np.linalg.norm(a - b))
    if anchor_dist < 1:
        anchor_dist = 1.0

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

        # No occlusion-box terms here.
        heuristic_raw = (
            1.00 * edge_mean
            + 0.70 * edge_min
            + 1.25 * anat
            + 0.20 * local_edge
            + 0.15 * min(ac, bc)
        )
        raw_scores.append(heuristic_raw)

        # label uses GT for training only
        tx = safe_float(row.get("target_orig_x", np.nan))
        ty = safe_float(row.get("target_orig_y", np.nan))
        cand_err = point_error(c[0], c[1], tx, ty)

        # soft target: close candidates get larger labels
        sigma = max(0.12 * anchor_dist, 8.0)
        quality = float(np.exp(-cand_err / sigma)) if np.isfinite(cand_err) else 0.0

        width_norm = max(float(w), 1.0)
        height_norm = max(float(h), 1.0)

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

            "candidate_x_norm": float(c[0]) / width_norm,
            "candidate_y_norm": float(c[1]) / height_norm,
            "anchor1_x_norm": float(a[0]) / width_norm,
            "anchor1_y_norm": float(a[1]) / height_norm,
            "anchor2_x_norm": float(b[0]) / width_norm,
            "anchor2_y_norm": float(b[1]) / height_norm,

            "heuristic_raw": heuristic_raw,

            "target_id_0": 1.0 if target_id == 0 else 0.0,
            "target_id_1": 1.0 if target_id == 1 else 0.0,
            "target_id_2": 1.0 if target_id == 2 else 0.0,
            "target_id_3": 1.0 if target_id == 3 else 0.0,
        }

        # Baseline columns if available. They are saved for comparison, not used as features.
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
        candidates=np.array([[r["candidate_x"], r["candidate_y"]] for r in records], dtype=np.float32),
        raw_scores=raw_scores,
        anchor_dist=anchor_dist,
    )

    for r, s in zip(records, slime_scores):
        r["heuristic_slime_score"] = float(s)

    return pd.DataFrame(records)


NO_OCC_FEATURES = [
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

    "target_id_0",
    "target_id_1",
    "target_id_2",
    "target_id_3",
]


def split_by_image(df, val_ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)

    names = df["image_name"].fillna(df["occluded_image_name"]).unique()
    rng.shuffle(names)

    n_val = max(1, int(len(names) * val_ratio))
    val_names = set(names[:n_val])

    val = df[df["image_name"].fillna(df["occluded_image_name"]).isin(val_names)].copy()
    train = df[~df["image_name"].fillna(df["occluded_image_name"]).isin(val_names)].copy()

    return train.reset_index(drop=True), val.reset_index(drop=True)


def summarize_best(best_df, method_col="learned_error_px"):
    out = {
        "count": len(best_df),
        "learned_mean_px": best_df[method_col].mean(),
        "learned_median_px": best_df[method_col].median(),
        "heuristic_mean_px": best_df["heuristic_error_px"].mean() if "heuristic_error_px" in best_df.columns else np.nan,
        "oracle_mean_px": best_df["oracle_error_px"].mean() if "oracle_error_px" in best_df.columns else np.nan,
    }

    if "yolo_target_error_px" in best_df.columns:
        out["yolo_mean_px"] = best_df["yolo_target_error_px"].mean()
        out["learned_win_vs_yolo"] = (best_df[method_col] < best_df["yolo_target_error_px"]).mean()

    if "prior_error_px" in best_df.columns:
        out["prior_mean_px"] = best_df["prior_error_px"].mean()

    return out
