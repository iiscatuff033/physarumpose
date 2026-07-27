import numpy as np
import cv2
import pandas as pd

TARGET_TO_COCO = {
    'right_elbow': 8,
    'left_elbow': 7,
    'right_knee': 14,
    'left_knee': 13,
}

ANCHOR_COCO = {
    'right_elbow': (6, 10),
    'left_elbow': (5, 9),
    'right_knee': (12, 16),
    'left_knee': (11, 15),
}

ANCHOR_NAMES = {
    'right_elbow': ('right_shoulder', 'right_wrist'),
    'left_elbow': ('left_shoulder', 'left_wrist'),
    'right_knee': ('right_hip', 'right_ankle'),
    'left_knee': ('left_hip', 'left_ankle'),
}


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


def get_anchor_points(row):
    target_name = row['target_name']
    a_id, b_id = ANCHOR_COCO[target_name]

    ax = safe_float(row.get(f'coco{a_id}_x', np.nan))
    ay = safe_float(row.get(f'coco{a_id}_y', np.nan))
    ac = safe_float(row.get(f'coco{a_id}_conf', 0.0), 0.0)

    bx = safe_float(row.get(f'coco{b_id}_x', np.nan))
    by = safe_float(row.get(f'coco{b_id}_y', np.nan))
    bc = safe_float(row.get(f'coco{b_id}_conf', 0.0), 0.0)

    return np.array([ax, ay], dtype=np.float32), np.array([bx, by], dtype=np.float32), ac, bc


def is_valid_point(p):
    return np.isfinite(p).all() and p[0] > 0 and p[1] > 0


def get_occlusion_box(row):
    x1 = safe_float(row.get('occ_x1', np.nan))
    y1 = safe_float(row.get('occ_y1', np.nan))
    x2 = safe_float(row.get('occ_x2', np.nan))
    y2 = safe_float(row.get('occ_y2', np.nan))
    if any(np.isnan(v) for v in [x1, y1, x2, y2]):
        return None
    return float(x1), float(y1), float(x2), float(y2)


def expand_box(box, margin):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return x1 - margin, y1 - margin, x2 + margin, y2 + margin


def inside_box_xy(x, y, box):
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def distance_to_box_center(x, y, box):
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return float(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))


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
        # reject clearly outside points first
        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        if ignore_box is not None and inside_box_xy(x, y, ignore_box):
            continue

        # round can produce w or h, so clip after rounding
        xi = int(round(x))
        yi = int(round(y))

        xi = max(0, min(w - 1, xi))
        yi = max(0, min(h - 1, yi))

        values.append(edge_map[yi, xi])

    if len(values) == 0:
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
    cosv = np.dot(v1, v2) / (n1 * n2)
    cosv = np.clip(cosv, -1.0, 1.0)
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
    d = float(np.linalg.norm(a - b)) + 1e-6

    min_ratio = min(d1, d2) / max(d1 + d2, 1e-6)
    balance_score = np.clip(min_ratio / 0.25, 0.0, 1.0)

    path_ratio = (d1 + d2) / d
    if path_ratio <= 1.8:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - (path_ratio - 1.8) / 1.2)

    return float(0.45 * angle_score + 0.35 * balance_score + 0.20 * length_score)


def occlusion_score(c, box, anchor_dist):
    if box is None:
        return 0.5
    x, y = float(c[0]), float(c[1])
    if inside_box_xy(x, y, box):
        return 1.0
    d = distance_to_box_center(x, y, box)
    x1, y1, x2, y2 = box
    diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    denom = max(0.50 * diag, 0.15 * anchor_dist, 1.0)
    return float(np.exp(-d / denom))


def prior_agreement_score(row, c, anchor_dist):
    px = safe_float(row.get('prior_pred_x', np.nan))
    py = safe_float(row.get('prior_pred_y', np.nan))
    if np.isnan(px) or np.isnan(py) or px <= 0 or py <= 0:
        return 0.5
    d = np.sqrt((float(c[0]) - px) ** 2 + (float(c[1]) - py) ** 2)
    denom = max(0.40 * anchor_dist, 1.0)
    return float(np.exp(-d / denom))


def generate_candidates(a, b, image_shape, box=None, grid_alphas=None, grid_betas=None):
    h, w = image_shape[:2]
    if grid_alphas is None:
        grid_alphas = np.linspace(0.25, 0.75, 11)
    if grid_betas is None:
        grid_betas = np.linspace(-0.70, 0.70, 17)

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

    if box is not None:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        for ox in [-0.35, 0.0, 0.35]:
            for oy in [-0.35, 0.0, 0.35]:
                x = cx + ox * bw
                y = cy + oy * bh
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


def score_candidates(row, image_bgr, candidates, a, b, weights):
    edge_map = make_edge_map(
        image_bgr,
        blur=int(weights.get('edge_blur', 3)),
        low=int(weights.get('edge_low', 50)),
        high=int(weights.get('edge_high', 150)),
    )
    box = get_occlusion_box(row)
    ignore_box = expand_box(box, margin=4) if box is not None else None
    anchor_dist = float(np.linalg.norm(a - b))
    records = []
    for idx, c in enumerate(candidates):
        edge_s = path_edge_score(edge_map, a, c, b, ignore_box=ignore_box)
        anatomy_s = anatomy_score(a, c, b)
        occ_s = occlusion_score(c, box, anchor_dist)
        prior_s = prior_agreement_score(row, c, anchor_dist)
        raw = (
            float(weights.get('edge_w', 1.0)) * edge_s
            + float(weights.get('anatomy_w', 1.0)) * anatomy_s
            + float(weights.get('occ_w', 1.2)) * occ_s
            + float(weights.get('prior_w', 0.35)) * prior_s
        )
        records.append({
            'candidate_id': idx,
            'x': float(c[0]),
            'y': float(c[1]),
            'edge_score': edge_s,
            'anatomy_score': anatomy_s,
            'occ_score': occ_s,
            'prior_score': prior_s,
            'raw_score': raw,
        })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    raw_scores = df['raw_score'].values.astype(np.float32)
    df['slime_score'] = slime_reinforce(
        candidates,
        raw_scores,
        anchor_dist=anchor_dist,
        iterations=int(weights.get('slime_iters', 8)),
        diffusion=float(weights.get('slime_diffusion', 0.20)),
        decay=float(weights.get('slime_decay', 0.08)),
    )
    return df


def choose_best_candidate(candidate_df):
    if candidate_df is None or len(candidate_df) == 0:
        return None
    return candidate_df.sort_values('slime_score', ascending=False).iloc[0]


def get_yolo_target(row):
    return safe_float(row.get('yolo_target_x', np.nan)), safe_float(row.get('yolo_target_y', np.nan))


def get_prior_target(row):
    return safe_float(row.get('prior_pred_x', np.nan)), safe_float(row.get('prior_pred_y', np.nan))


def get_geometry_target(row):
    return safe_float(row.get('geometry_x', np.nan)), safe_float(row.get('geometry_y', np.nan))
