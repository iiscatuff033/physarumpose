import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TARGET_TO_TASK = {
    "right_elbow": 0,
    "left_elbow": 1,
    "right_knee": 2,
    "left_knee": 3,
}

ANCHOR_NAMES = {
    "right_elbow": ("right_shoulder", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_wrist"),
    "right_knee": ("right_hip", "right_ankle"),
    "left_knee": ("left_hip", "left_ankle"),
}

FEATURE_NAMES = [
    "cand_x_norm",
    "cand_y_norm",
    "alpha",
    "beta",
    "anchor_dist_norm",
    "anchor1_conf",
    "anchor2_conf",
    "anchor_min_conf",
    "d1_ratio",
    "d2_ratio",
    "path_ratio",
    "angle_norm",
    "edge_score",
    "anatomy_score",
    "occ_score",
    "inside_occ",
    "dist_occ_norm",
    "raw_heuristic_score",
    "slime_reinforced_score",
    "target_right_elbow",
    "target_left_elbow",
    "target_right_knee",
    "target_left_knee",
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


def get_anchor_value(row, anchor_name, suffix):
    """
    Supports both column styles:
    - anchor_pred_right_shoulder_x
    - anchor_right_shoulder_x
    """
    candidates = [
        f"anchor_pred_{anchor_name}_{suffix}",
        f"anchor_{anchor_name}_{suffix}",
    ]
    for c in candidates:
        if c in row.index:
            return safe_float(row.get(c, np.nan))
    return np.nan


def get_anchors_from_anchornet(row):
    target_name = row["target_name"]
    a_name, b_name = ANCHOR_NAMES[target_name]

    ax = get_anchor_value(row, a_name, "x")
    ay = get_anchor_value(row, a_name, "y")
    ac = get_anchor_value(row, a_name, "conf")

    bx = get_anchor_value(row, b_name, "x")
    by = get_anchor_value(row, b_name, "y")
    bc = get_anchor_value(row, b_name, "conf")

    a = np.array([ax, ay], dtype=np.float32)
    b = np.array([bx, by], dtype=np.float32)

    return a_name, b_name, a, b, float(ac), float(bc)


def valid_point(p):
    return np.isfinite(p).all() and p[0] > 0 and p[1] > 0


def get_occlusion_box(row):
    keys = ["occ_x1", "occ_y1", "occ_x2", "occ_y2"]
    vals = [safe_float(row.get(k, np.nan)) for k in keys]
    if any(np.isnan(v) for v in vals):
        return None
    return tuple(float(v) for v in vals)


def inside_box(x, y, box):
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_box(box, margin):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return x1 - margin, y1 - margin, x2 + margin, y2 + margin


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


def edge_support_along_segment(edge_map, p1, p2, ignore_box=None, n=80):
    h, w = edge_map.shape[:2]
    pts = sample_line_points(p1, p2, n=n)
    values = []

    for x, y in pts:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if ignore_box is not None and inside_box(x, y, ignore_box):
            continue
        xi = int(round(x))
        yi = int(round(y))
        xi = max(0, min(w - 1, xi))
        yi = max(0, min(h - 1, yi))
        values.append(edge_map[yi, xi])

    if not values:
        return 0.0
    return float(np.mean(values))


def path_edge_score(edge_map, a, c, b, ignore_box=None):
    s1 = edge_support_along_segment(edge_map, a, c, ignore_box=ignore_box)
    s2 = edge_support_along_segment(edge_map, c, b, ignore_box=ignore_box)
    return float((s1 + s2) / 2.0)


def angle_at_candidate(a, c, b):
    v1 = a - c
    v2 = b - c
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def anatomy_score_and_parts(a, c, b):
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

    score = float(0.45 * angle_score + 0.35 * balance_score + 0.20 * length_score)
    return score, angle, d1, d2, path_ratio


def occlusion_features(c, box, anchor_dist):
    if box is None:
        return 0.5, 0.0, 1.0
    x, y = float(c[0]), float(c[1])
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    d = float(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))
    diag = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    denom = max(0.50 * diag, 0.15 * anchor_dist, 1.0)
    occ_score = float(np.exp(-d / denom))
    return occ_score, float(inside_box(x, y, box)), float(d / max(anchor_dist, 1.0))


def generate_candidates(a, b, image_shape, box=None, n_alpha=13, n_beta=21):
    h, w = image_shape[:2]
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    v = b - a
    d = float(np.linalg.norm(v))
    if d < 5:
        return np.zeros((0, 2), dtype=np.float32)

    unit = v / d
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    alphas = np.linspace(0.18, 0.82, n_alpha)
    betas = np.linspace(-0.85, 0.85, n_beta)

    out = []
    for alpha in alphas:
        base = a + alpha * v
        for beta in betas:
            p = base + beta * d * perp
            x, y = float(p[0]), float(p[1])
            if 0 <= x < w and 0 <= y < h:
                out.append([x, y])

    if box is not None:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        for ox in [-0.50, -0.25, 0.0, 0.25, 0.50]:
            for oy in [-0.50, -0.25, 0.0, 0.25, 0.50]:
                x = cx + ox * bw
                y = cy + oy * bh
                if 0 <= x < w and 0 <= y < h:
                    out.append([x, y])

    if not out:
        return np.zeros((0, 2), dtype=np.float32)

    arr = np.array(out, dtype=np.float32)
    rounded = np.round(arr / 3.0).astype(np.int32)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return arr[np.sort(idx)]


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
        mx = float(flow.max())
        if mx > 0:
            flow = flow / mx
    return flow


def candidate_alpha_beta(a, b, c):
    v = b - a
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return 0.5, 0.0
    unit = v / d
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    rel = c - a
    alpha = float(np.dot(rel, unit) / d)
    beta = float(np.dot(rel, perp) / d)
    return alpha, beta


def target_one_hot(target_name):
    arr = np.zeros(4, dtype=np.float32)
    arr[TARGET_TO_TASK[target_name]] = 1.0
    return arr


def build_candidate_table_for_row(row, image_bgr, sample_id, use_edges=True):
    a_name, b_name, a, b, ac, bc = get_anchors_from_anchornet(row)
    box = get_occlusion_box(row)

    if not valid_point(a) or not valid_point(b):
        return pd.DataFrame(), "bad_anchors"

    if not np.isfinite(ac) or not np.isfinite(bc):
        return pd.DataFrame(), "bad_anchor_conf"

    candidates = generate_candidates(a, b, image_bgr.shape, box=box)
    if len(candidates) == 0:
        return pd.DataFrame(), "no_candidates"

    h, w = image_bgr.shape[:2]
    image_diag = float(np.sqrt(w * w + h * h))
    anchor_dist = float(np.linalg.norm(a - b))

    if use_edges:
        edge_map = make_edge_map(image_bgr)
        ignore_box = expand_box(box, margin=4)
    else:
        edge_map = None
        ignore_box = None

    raw_scores = []
    records = []

    for cid, c in enumerate(candidates):
        edge_s = path_edge_score(edge_map, a, c, b, ignore_box=ignore_box) if use_edges else 0.0
        anatomy_s, angle, d1, d2, path_ratio = anatomy_score_and_parts(a, c, b)
        occ_s, inside_occ, dist_occ_norm = occlusion_features(c, box, anchor_dist)

        raw = 1.0 * edge_s + 1.0 * anatomy_s + 1.2 * occ_s
        raw_scores.append(raw)

        alpha, beta = candidate_alpha_beta(a, b, c)
        d1_ratio = d1 / max(anchor_dist, 1.0)
        d2_ratio = d2 / max(anchor_dist, 1.0)

        records.append({
            "sample_id": int(sample_id),
            "candidate_id": int(cid),
            "image_name": row.get("image_name", ""),
            "occluded_image_name": row.get("occluded_image_name", ""),
            "target_name": row["target_name"],
            "target_orig_x": safe_float(row.get("target_orig_x")),
            "target_orig_y": safe_float(row.get("target_orig_y")),
            "occ_x1": safe_float(row.get("occ_x1")),
            "occ_y1": safe_float(row.get("occ_y1")),
            "occ_x2": safe_float(row.get("occ_x2")),
            "occ_y2": safe_float(row.get("occ_y2")),
            "anchor1_name": a_name,
            "anchor2_name": b_name,
            "anchor1_x": float(a[0]),
            "anchor1_y": float(a[1]),
            "anchor2_x": float(b[0]),
            "anchor2_y": float(b[1]),
            "candidate_x": float(c[0]),
            "candidate_y": float(c[1]),
            "yolo_target_error_px": safe_float(row.get("yolo_target_error_px", np.nan)),
            "yolo_target_x": safe_float(row.get("yolo_target_x", np.nan)),
            "yolo_target_y": safe_float(row.get("yolo_target_y", np.nan)),
            "cand_x_norm": float(c[0] / max(w, 1)),
            "cand_y_norm": float(c[1] / max(h, 1)),
            "alpha": alpha,
            "beta": beta,
            "anchor_dist_norm": float(anchor_dist / max(image_diag, 1.0)),
            "anchor1_conf": float(ac),
            "anchor2_conf": float(bc),
            "anchor_min_conf": float(min(ac, bc)),
            "d1_ratio": float(d1_ratio),
            "d2_ratio": float(d2_ratio),
            "path_ratio": float(path_ratio),
            "angle_norm": float(angle / 180.0),
            "edge_score": float(edge_s),
            "anatomy_score": float(anatomy_s),
            "occ_score": float(occ_s),
            "inside_occ": float(inside_occ),
            "dist_occ_norm": float(dist_occ_norm),
            "raw_heuristic_score": float(raw),
            "slime_reinforced_score": 0.0,
        })

    if not records:
        return pd.DataFrame(), "no_records"

    arr = np.array(raw_scores, dtype=np.float32)
    reinforced = slime_reinforce(candidates, arr, anchor_dist=anchor_dist)
    for i, val in enumerate(reinforced):
        records[i]["slime_reinforced_score"] = float(val)

    gt = np.array([safe_float(row.get("target_orig_x")), safe_float(row.get("target_orig_y"))], dtype=np.float32)
    label_sigma = max(0.12 * anchor_dist, 12.0)
    good_radius = max(0.10 * anchor_dist, 14.0)

    for rec in records:
        c = np.array([rec["candidate_x"], rec["candidate_y"]], dtype=np.float32)
        dist = float(np.linalg.norm(c - gt))
        rec["candidate_error_px"] = dist
        rec["label_quality"] = float(np.exp(-dist / label_sigma))
        rec["label_good"] = float(dist <= good_radius)
        oh = target_one_hot(row["target_name"])
        rec["target_right_elbow"] = float(oh[0])
        rec["target_left_elbow"] = float(oh[1])
        rec["target_right_knee"] = float(oh[2])
        rec["target_left_knee"] = float(oh[3])

    df = pd.DataFrame(records)
    best_idx = df["candidate_error_px"].idxmin()
    df["is_best_candidate"] = 0.0
    df.loc[best_idx, "is_best_candidate"] = 1.0
    return df, "ok"


def summarize_predictions(df, pred_col="learned_error_px"):
    out = {
        "count": len(df),
        "learned_mean_px": df[pred_col].mean(),
        "learned_median_px": df[pred_col].median(),
    }
    if "yolo_target_error_px" in df.columns:
        out["yolo_mean_px"] = df["yolo_target_error_px"].mean()
        out["yolo_median_px"] = df["yolo_target_error_px"].median()
        out["learned_win_vs_yolo"] = (df[pred_col] < df["yolo_target_error_px"]).mean()
    if "heuristic_error_px" in df.columns:
        out["heuristic_mean_px"] = df["heuristic_error_px"].mean()
        out["heuristic_median_px"] = df["heuristic_error_px"].median()
        out["learned_win_vs_heuristic"] = (df[pred_col] < df["heuristic_error_px"]).mean()
    return out
